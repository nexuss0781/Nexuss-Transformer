"""
Evaluation Metrics for Nexuss Transformer Framework
Perplexity, accuracy, and custom LLM benchmarks
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import math


@dataclass
class EvaluationResults:
    """Container for evaluation metrics"""
    
    # Core metrics
    perplexity: float = 0.0
    loss: float = 0.0
    accuracy: float = 0.0
    
    # Token-level metrics
    token_accuracy: float = 0.0
    exact_match: float = 0.0
    
    # Generation metrics (for generation tasks)
    bleu_score: float = 0.0
    rouge_l: float = 0.0
    
    # Task-specific metrics
    task_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Summary
    num_samples: int = 0
    num_tokens: int = 0


def compute_perplexity(model, dataloader: DataLoader, device: torch.device) -> float:
    """Compute perplexity on a dataset"""
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            labels = batch.get("labels", input_ids) if isinstance(batch, dict) else input_ids
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Count tokens (excluding padding)
            if isinstance(batch, dict) and "attention_mask" in batch:
                num_tokens = batch["attention_mask"].sum().item()
            else:
                num_tokens = input_ids.numel()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)
    
    return perplexity


def compute_accuracy(model, dataloader: DataLoader, device: torch.device) -> float:
    """Compute token-level accuracy"""
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            labels = batch.get("labels", input_ids) if isinstance(batch, dict) else input_ids
            
            outputs = model(input_ids=input_ids)
            predictions = outputs.logits.argmax(dim=-1)
            
            # Shift for causal LM (predict next token)
            pred_shifted = predictions[:, :-1]
            label_shifted = labels[:, 1:]
            
            # Mask out padding if present
            if isinstance(batch, dict) and "attention_mask" in batch:
                mask = batch["attention_mask"][:, 1:]
                correct += (pred_shifted[mask == 1] == label_shifted[mask == 1]).sum().item()
                total += mask.sum().item()
            else:
                correct += (pred_shifted == label_shifted).sum().item()
                total += label_shifted.numel()
    
    return correct / max(total, 1)


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    compute_generation_metrics: bool = False,
    tokenizer=None,
) -> EvaluationResults:
    """Comprehensive model evaluation"""
    
    model.eval()
    results = EvaluationResults()
    
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            labels = batch.get("labels", input_ids) if isinstance(batch, dict) else input_ids
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Count tokens
            if isinstance(batch, dict) and "attention_mask" in batch:
                num_tokens = batch["attention_mask"].sum().item()
            else:
                num_tokens = input_ids.numel()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Compute accuracy
            predictions = outputs.logits.argmax(dim=-1)
            pred_shifted = predictions[:, :-1]
            label_shifted = labels[:, 1:]
            
            if isinstance(batch, dict) and "attention_mask" in batch:
                mask = batch["attention_mask"][:, 1:]
                correct_tokens += (pred_shifted[mask == 1] == label_shifted[mask == 1]).sum().item()
            else:
                correct_tokens += (pred_shifted == label_shifted).sum().item()
            
            results.num_samples += input_ids.size(0)
    
    # Compute final metrics
    results.num_tokens = total_tokens
    results.loss = total_loss / max(total_tokens, 1)
    results.perplexity = math.exp(results.loss)
    results.token_accuracy = correct_tokens / max(total_tokens - results.num_samples, 1)
    results.accuracy = results.token_accuracy
    
    # Optional: generation metrics
    if compute_generation_metrics and tokenizer is not None:
        gen_results = compute_generation_metrics_batch(model, dataloader, tokenizer, device)
        results.bleu_score = gen_results.get("bleu", 0.0)
        results.rouge_l = gen_results.get("rouge_l", 0.0)
    
    return results


def compute_generation_metrics_batch(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    """Compute generation-based metrics (BLEU, ROUGE)"""
    
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer
    except ImportError:
        # Return zeros if libraries not installed
        return {"bleu": 0.0, "rouge_l": 0.0}
    
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch.to(device)
            
            # Generate
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            
            # Decode
            for i in range(input_ids.size(0)):
                # Reference (ground truth continuation)
                if isinstance(batch, dict) and "labels" in batch:
                    ref_ids = batch["labels"][i]
                    ref_text = tokenizer.decode(ref_ids[ref_ids != -100], skip_special_tokens=True)
                    references.append(ref_text)
                
                # Hypothesis (generated)
                hyp_ids = outputs[i]
                hyp_text = tokenizer.decode(hyp_ids, skip_special_tokens=True)
                hypotheses.append(hyp_text)
    
    # Compute BLEU
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        
        if len(ref_tokens) > 0 and len(hyp_tokens) > 0:
            score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
            bleu_scores.append(score)
    
    avg_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
    
    # Compute ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge_l = sum(rouge_scores) / max(len(rouge_scores), 1)
    
    return {"bleu": avg_bleu, "rouge_l": avg_rouge_l}


def benchmark_throughput(
    model,
    tokenizer,
    device: torch.device,
    sequence_length: int = 512,
    batch_size: int = 1,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Benchmark model throughput (tokens/second)"""
    
    model.eval()
    
    # Create dummy input
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, sequence_length)).to(device)
    
    # Warmup
    with torch.no_grad():
        _ = model(input_ids)
    
    # Benchmark prefill (prompt processing)
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    
    if device.type == "cuda":
        start.record()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_ids)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        import time
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(input_ids)
        elapsed_ms = (time.time() - start_time) * 1000
    
    tokens_processed = batch_size * sequence_length * num_iterations
    prefill_throughput = tokens_processed / (elapsed_ms / 1000)
    
    # Benchmark decoding (generation)
    max_new_tokens = 64
    
    if device.type == "cuda":
        start.record()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model.generate(input_ids[:, :32], max_new_tokens=max_new_tokens, do_sample=False)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        import time
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model.generate(input_ids[:, :32], max_new_tokens=max_new_tokens, do_sample=False)
        elapsed_ms = (time.time() - start_time) * 1000
    
    tokens_generated = batch_size * max_new_tokens * num_iterations
    decode_throughput = tokens_generated / (elapsed_ms / 1000)
    
    return {
        "prefill_throughput": prefill_throughput,  # tokens/sec
        "decode_throughput": decode_throughput,     # tokens/sec
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


def compare_models(
    model_a,
    model_b,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Compare two models on the same dataset"""
    
    results_a = evaluate_model(model_a, dataloader, device)
    results_b = evaluate_model(model_b, dataloader, device)
    
    comparison = {
        "model_a": {
            "perplexity": results_a.perplexity,
            "loss": results_a.loss,
            "accuracy": results_a.accuracy,
        },
        "model_b": {
            "perplexity": results_b.perplexity,
            "loss": results_b.loss,
            "accuracy": results_b.accuracy,
        },
        "improvement": {
            "perplexity": ((results_a.perplexity - results_b.perplexity) / results_a.perplexity) * 100,
            "loss": ((results_a.loss - results_b.loss) / results_a.loss) * 100,
            "accuracy": ((results_b.accuracy - results_a.accuracy) / results_a.accuracy) * 100,
        },
    }
    
    return comparison
