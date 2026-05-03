# Tutorial 07: Model Validation, Testing & Quality Assurance

## Overview

This tutorial covers comprehensive model validation, testing strategies, and quality assurance processes essential for production-ready AI systems. We'll explore NTF's unified metrics utilities, statistical validation methods, bias detection, robustness testing, and systematic evaluation frameworks.

## Table of Contents

1. [Validation Fundamentals](#validation-fundamentals)
2. [NTF Metrics Utilities](#ntf-metrics-utilities)
3. [Train/Validation/Test Splits](#trainvalidationtest-splits)
4. [Cross-Validation Techniques](#cross-validation-techniques)
5. [Statistical Significance Testing](#statistical-significance-testing)
6. [Bias and Fairness Detection](#bias-and-fairness-detection)
7. [Robustness Testing](#robustness-testing)
8. [Adversarial Testing](#adversarial-testing)
9. [Domain Shift Detection](#domain-shift-detection)
10. [Calibration and Confidence Estimation](#calibration-and-confidence-estimation)
11. [A/B Testing Framework](#ab-testing-framework)
12. [Regression Testing for Models](#regression-testing-for-models)
13. [Quality Gates and Release Criteria](#quality-gates-and-release-criteria)

---

## Validation Fundamentals

### Why Validation Matters

Validation ensures your model:
- Generalizes to unseen data
- Doesn't overfit training distributions
- Meets performance requirements
- Behaves safely across edge cases
- Maintains consistency across versions

### Validation Pyramid

```
                    Production Monitoring
                           /\
                          /  \
                         /    \
                        /------\
                       / A/B    \
                      / Testing  \
                     /------------\
                    / Holdout      \
                   /   Testing      \
                  /------------------\
                 / Cross-Validation   \
                /----------------------\
               /  Train/Val Split      \
              /--------------------------\
```

**Key Principles:**
1. **Data Isolation**: Never leak test data into training
2. **Distribution Matching**: Test data should match production distribution
3. **Statistical Power**: Ensure sufficient sample sizes
4. **Multiple Metrics**: Evaluate across diverse dimensions
5. **Reproducibility**: Fixed seeds and documented procedures

---

## NTF Metrics Utilities

### Using NTF's Unified Evaluation Interface

NTF provides comprehensive metrics utilities through `ntf.utils.metrics`. This replaces manual metric implementations with a unified, efficient interface.

```python
from ntf.utils.metrics import (
    compute_perplexity,
    compute_accuracy,
    evaluate_model,
    compare_models,
    benchmark_throughput,
    EvaluationResults
)
from torch.utils.data import DataLoader
import torch

# Load your model and tokenizer
model, tokenizer = load_model_and_tokenizer("path/to/model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare test dataloader
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Comprehensive evaluation
results = evaluate_model(
    model=model,
    dataloader=test_dataloader,
    device=device,
    compute_generation_metrics=True,
    tokenizer=tokenizer
)

print(f"Perplexity: {results.perplexity:.2f}")
print(f"Loss: {results.loss:.4f}")
print(f"Token Accuracy: {results.token_accuracy:.4f}")
print(f"BLEU Score: {results.bleu_score:.4f}")
print(f"ROUGE-L: {results.rouge_l:.4f}")
```

### Individual Metric Functions

For specific metric computation, NTF provides standalone functions:

```python
# Compute perplexity only
perplexity = compute_perplexity(model, test_dataloader, device)
print(f"Perplexity: {perplexity:.2f}")

# Compute accuracy only
accuracy = compute_accuracy(model, test_dataloader, device)
print(f"Accuracy: {accuracy:.4f}")
```

### Comparing Multiple Checkpoints

NTF makes it easy to compare different model checkpoints:

```python
from ntf.utils.metrics import compare_models

# Compare two model versions
comparison = compare_models(
    model_a=model_v1,
    model_b=model_v2,
    dataloader=val_dataloader,
    device=device
)

print(f"Model A Perplexity: {comparison['model_a']['perplexity']:.2f}")
print(f"Model B Perplexity: {comparison['model_b']['perplexity']:.2f}")
print(f"Improvement: {comparison['improvement']['perplexity']:.2f}%")
print(f"Accuracy Gain: {comparison['improvement']['accuracy']:.2f}%")
```

### Benchmarking Throughput

For production deployment, benchmark model throughput:

```python
throughput_results = benchmark_throughput(
    model=model,
    tokenizer=tokenizer,
    device=device,
    sequence_length=512,
    batch_size=1,
    num_iterations=10
)

print(f"Prefill Throughput: {throughput_results['prefill_throughput']:.2f} tokens/sec")
print(f"Decode Throughput: {throughput_results['decode_throughput']:.2f} tokens/sec")
```

### Metric Selection Guide

Different tasks require different evaluation metrics. Use this guide to select appropriate metrics:

| Task Type | Recommended Metrics | NTF Functions |
|-----------|---------------------|---------------|
| Text Generation | Perplexity, BLEU, ROUGE, BERTScore | `evaluate_model(compute_generation_metrics=True)` |
| Classification | Accuracy, F1, Precision, Recall | `compute_accuracy()` + custom F1 |
| Summarization | ROUGE, BERTScore | `evaluate_model()` with ROUGE |
| Translation | BLEU, chrF, COMET | `evaluate_model()` with BLEU |
| Question Answering | Exact Match, F1 | Custom implementation |
| Language Modeling | Perplexity | `compute_perplexity()` |

### Checkpoint Comparison Workflow

Here's a complete workflow for comparing multiple checkpoints during development:

```python
from ntf.utils.metrics import evaluate_model
from pathlib import Path
import json

def compare_checkpoints(checkpoint_paths, eval_dataset, tokenizer, device):
    """Compare multiple checkpoints on the same evaluation dataset."""
    
    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    results = {}
    
    for checkpoint_path in checkpoint_paths:
        print(f"\nEvaluating {checkpoint_path}...")
        
        # Load checkpoint
        model, _ = load_model_and_tokenizer(checkpoint_path)
        model.to(device)
        model.eval()
        
        # Evaluate
        eval_results = evaluate_model(
            model=model,
            dataloader=eval_dataloader,
            device=device,
            compute_generation_metrics=True,
            tokenizer=tokenizer
        )
        
        # Store results
        results[checkpoint_path] = {
            'perplexity': eval_results.perplexity,
            'loss': eval_results.loss,
            'accuracy': eval_results.accuracy,
            'bleu': eval_results.bleu_score,
            'rouge_l': eval_results.rouge_l
        }
        
        print(f"  Perplexity: {eval_results.perplexity:.2f}")
        print(f"  Accuracy: {eval_results.accuracy:.4f}")
    
    # Find best checkpoint
    best_checkpoint = min(results.keys(), key=lambda k: results[k]['perplexity'])
    print(f"\nBest checkpoint (lowest perplexity): {best_checkpoint}")
    
    return results

# Usage
checkpoints = [
    "./checkpoints/step_1000",
    "./checkpoints/step_2000",
    "./checkpoints/step_3000",
    "./checkpoints/final"
]

all_results = compare_checkpoints(
    checkpoints, 
    val_dataset, 
    tokenizer, 
    device
)

# Save results for tracking
with open("./evaluation_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
```

---

## Train/Validation/Test Splits

### Strategic Data Partitioning

```python
from sklearn.model_selection import train_test_split
import numpy as np

def strategic_data_split(data, labels, strategy='stratified'):
    """
    Create train/val/test splits with proper stratification.
    
    Args:
        data: Input samples
        labels: Corresponding labels
        strategy: 'stratified', 'temporal', 'grouped'
    
    Returns:
        train, val, test splits
    """
    if strategy == 'stratified':
        # First split: train+val vs test (80/20)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data, labels, 
            test_size=0.2, 
            stratify=labels,  # Maintain class distribution
            random_state=42
        )
        
        # Second split: train vs val (80/20 of remaining = 64/16 total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=0.2,
            stratify=y_train_val,
            random_state=42
        )
        
    elif strategy == 'temporal':
        # Time-based split for temporal data
        split_point_1 = int(len(data) * 0.6)
        split_point_2 = int(len(data) * 0.8)
        
        X_train = data[:split_point_1]
        X_val = data[split_point_1:split_point_2]
        X_test = data[split_point_2:]
        
        y_train = labels[:split_point_1]
        y_val = labels[split_point_1:split_point_2]
        y_test = labels[split_point_2:]
        
    elif strategy == 'grouped':
        # Group-based split (e.g., by user, document, session)
        from sklearn.model_selection import GroupShuffleSplit
        
        groups = get_groups(data)  # Your grouping logic
        
        gss = GroupShuffleSplit(
            n_splits=1, 
            test_size=0.2, 
            random_state=42
        )
        train_val_idx, test_idx = next(gss.split(data, groups=groups))
        
        X_train_val, X_test = data[train_val_idx], data[test_idx]
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]
        
        # Split train_val further
        gss2 = GroupShuffleSplit(
            n_splits=1, 
            test_size=0.2, 
            random_state=42
        )
        train_idx, val_idx = next(gss2.split(
            X_train_val, 
            groups=groups[train_val_idx]
        ))
        
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

# Usage example
splits = strategic_data_split(texts, labels, strategy='stratified')
print(f"Train: {len(splits['train'][0])}, Val: {len(splits['val'][0])}, Test: {len(splits['test'][0])}")
```

### Split Ratio Guidelines

| Dataset Size | Train | Val | Test | Rationale |
|-------------|-------|-----|------|-----------|
| < 10K | 70% | 15% | 15% | Need more validation signal |
| 10K-100K | 80% | 10% | 10% | Balanced approach |
| 100K-1M | 90% | 5% | 5% | Large data, less validation needed |
| > 1M | 95% | 2.5% | 2.5% | Massive data, small holdouts sufficient |

### Common Splitting Mistakes

❌ **Data Leakage**: 
```python
# WRONG: Preprocessing before split
scaler.fit(data)  # Fits on ALL data including test!
data_scaled = scaler.transform(data)
# Then split...

# CORRECT: Split first
X_train, X_test = train_test_split(data, test_size=0.2)
scaler.fit(X_train)  # Fit only on train
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Transform test with train stats
```

❌ **Temporal Leakage**:
```python
# WRONG: Random shuffle on time-series
random.shuffle(time_series_data)

# CORRECT: Respect temporal order
cutoff = int(len(data) * 0.8)
train = data[:cutoff]
test = data[cutoff:]
```

---

## Cross-Validation Techniques

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch.utils.data import DataLoader

def k_fold_cross_validation(model_class, config, data, labels, k=5):
    """
    Perform k-fold cross-validation for robust performance estimation.
    
    Returns metrics for each fold and aggregate statistics.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{k}")
        print(f"{'='*50}")
        
        # Prepare fold data
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Initialize fresh model for this fold
        model = model_class(config)
        
        # Train on this fold
        trainer = Trainer(
            model=model,
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            config=config
        )
        
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)
        
        print(f"Fold {fold + 1} Accuracy: {metrics['accuracy']:.4f}")
        print(f"Fold {fold + 1} F1: {metrics['f1']:.4f}")
    
    # Aggregate results
    aggregated = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'all': values
        }
    
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*50}")
    for metric, stats in aggregated.items():
        print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    return aggregated, fold_metrics

# Usage
cv_results, fold_details = k_fold_cross_validation(
    MyModel, config, texts, labels, k=5
)
```

### Nested Cross-Validation for Hyperparameter Tuning

```python
from sklearn.model_selection import ParameterGrid

def nested_cross_validation(model_class, base_config, data, labels, 
                           param_grid, outer_k=5, inner_k=3):
    """
    Nested CV: Outer loop for evaluation, inner loop for hyperparameter selection.
    
    Prevents optimistic bias from hyperparameter tuning.
    """
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)
    outer_scores = []
    best_params_per_fold = []
    
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(data, labels)
    ):
        print(f"\n{'='*60}")
        print(f"OUTER FOLD {outer_fold + 1}/{outer_k}")
        print(f"{'='*60}")
        
        # Outer split
        X_outer_train = data[outer_train_idx]
        X_outer_test = data[outer_test_idx]
        y_outer_train = labels[outer_train_idx]
        y_outer_test = labels[outer_test_idx]
        
        # Inner CV for hyperparameter selection
        inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=42)
        
        param_scores = {params: [] for params in ParameterGrid(param_grid)}
        
        for inner_train_idx, inner_val_idx in inner_cv.split(
            X_outer_train, y_outer_train
        ):
            X_inner_train = X_outer_train[inner_train_idx]
            X_inner_val = X_outer_train[inner_val_idx]
            y_inner_train = y_outer_train[inner_train_idx]
            y_inner_val = y_outer_train[inner_val_idx]
            
            # Test each parameter combination
            for params in ParameterGrid(param_grid):
                config = {**base_config, **params}
                model = model_class(config)
                
                trainer = Trainer(
                    model=model,
                    train_data=(X_inner_train, y_inner_train),
                    val_data=(X_inner_val, y_inner_val),
                    config=config
                )
                
                trainer.train()
                metrics = trainer.evaluate()
                
                param_scores[params].append(metrics['accuracy'])
        
        # Select best parameters based on inner CV
        best_params = max(param_scores.keys(), 
                         key=lambda p: np.mean(param_scores[p]))
        best_score = np.mean(param_scores[best_params])
        
        print(f"Best params for outer fold {outer_fold + 1}: {best_params}")
        print(f"Inner CV score: {best_score:.4f}")
        
        # Train final model for this outer fold with best params
        final_config = {**base_config, **best_params}
        final_model = model_class(final_config)
        
        final_trainer = Trainer(
            model=final_model,
            train_data=(X_outer_train, y_outer_train),
            val_data=None,  # Use all outer train data
            config=final_config
        )
        
        final_trainer.train()
        
        # Evaluate on held-out outer test set
        test_metrics = final_trainer.evaluate_on_test(X_outer_test, y_outer_test)
        outer_scores.append(test_metrics['accuracy'])
        best_params_per_fold.append(best_params)
    
    # Final aggregated results
    print(f"\n{'='*60}")
    print("NESTED CV FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    print(f"Range: [{np.min(outer_scores):.4f}, {np.max(outer_scores):.4f}]")
    
    return {
        'mean_score': np.mean(outer_scores),
        'std_score': np.std(outer_scores),
        'scores': outer_scores,
        'best_params_per_fold': best_params_per_fold
    }

# Example usage
param_grid = {
    'learning_rate': [1e-5, 3e-5, 5e-5],
    'batch_size': [16, 32],
    'num_layers': [6, 12]
}

nested_results = nested_cross_validation(
    TransformerModel, 
    base_config, 
    data, 
    labels, 
    param_grid,
    outer_k=5,
    inner_k=3
)
```

### Leave-One-Out and Leave-P-Out

```python
from sklearn.model_selection import LeaveOneOut, LeavePOut

def leave_one_out_validation(model_class, config, data, labels, max_samples=1000):
    """
    Leave-One-Out CV: Extremely thorough but computationally expensive.
    Use only for small datasets (< 1000 samples).
    """
    if len(data) > max_samples:
        print(f"Warning: LOO is too expensive for {len(data)} samples.")
        print(f"Consider using k-fold with k=10 instead.")
        return None
    
    loo = LeaveOneOut()
    scores = []
    
    for train_idx, test_idx in loo.split(data):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        model = model_class(config)
        trainer = Trainer(model, (X_train, y_train), None, config)
        trainer.train()
        
        metrics = trainer.evaluate_on_test(X_test, y_test)
        scores.append(metrics['accuracy'])
    
    return {'mean': np.mean(scores), 'std': np.std(scores), 'all': scores}
```

---

## Statistical Significance Testing

### Comparing Two Models

```python
from scipy import stats
import numpy as np

def paired_t_test(model_a_predictions, model_b_predictions, ground_truth):
    """
    Paired t-test to compare two models on the same test set.
    
    Tests if the difference in performance is statistically significant.
    """
    # Calculate per-sample correctness
    correct_a = (model_a_predictions == ground_truth).astype(int)
    correct_b = (model_b_predictions == ground_truth).astype(int)
    
    # Paired t-test on correctness scores
    t_statistic, p_value = stats.ttest_rel(correct_a, correct_b)
    
    acc_a = np.mean(correct_a)
    acc_b = np.mean(correct_b)
    
    print(f"Model A Accuracy: {acc_a:.4f}")
    print(f"Model B Accuracy: {acc_b:.4f}")
    print(f"Difference: {acc_b - acc_a:.4f}")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        significance = "SIGNIFICANT" if acc_b > acc_a else "SIGNIFICANT (worse)"
        print(f"Result: Model B is {significance} than Model A (p < 0.05)")
    else:
        print(f"Result: No significant difference (p >= 0.05)")
    
    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'accuracy_difference': acc_b - acc_a
    }

def mcnemar_test(model_a_predictions, model_b_predictions, ground_truth):
    """
    McNemar's test for paired nominal data.
    More appropriate than t-test for classification accuracy comparison.
    """
    # Build contingency table
    both_correct = np.sum((model_a_predictions == ground_truth) & 
                         (model_b_predictions == ground_truth))
    a_correct_b_wrong = np.sum((model_a_predictions == ground_truth) & 
                               (model_b_predictions != ground_truth))
    a_wrong_b_correct = np.sum((model_a_predictions != ground_truth) & 
                               (model_b_predictions == ground_truth))
    both_wrong = np.sum((model_a_predictions != ground_truth) & 
                       (model_b_predictions != ground_truth))
    
    print("Contingency Table:")
    print(f"                Model B Correct | Model B Wrong")
    print(f"Model A Correct     {both_correct:6d}      {a_correct_b_wrong:6d}")
    print(f"Model A Wrong       {a_wrong_b_correct:6d}      {both_wrong:6d}")
    
    # McNemar's test statistic (with continuity correction)
    b = a_correct_b_wrong
    c = a_wrong_b_correct
    
    if b + c == 0:
        print("Cannot perform test: no discordant pairs")
        return None
    
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, 1)
    
    print(f"\nMcNemar's Chi-squared: {chi2:.4f}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        winner = "Model B" if c > b else "Model A"
        print(f"Result: {winner} is significantly better (p < 0.05)")
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'discordant_pairs': {'b': b, 'c': c}
    }

# Bootstrap confidence intervals
def bootstrap_confidence_interval(predictions, ground_truth, 
                                  metric_fn, n_bootstrap=1000, 
                                  confidence_level=0.95):
    """
    Estimate confidence intervals using bootstrapping.
    """
    n_samples = len(predictions)
    bootstrap_scores = []
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sampled_preds = predictions[indices]
        sampled_true = ground_truth[indices]
        
        score = metric_fn(sampled_preds, sampled_true)
        bootstrap_scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    mean_score = np.mean(bootstrap_scores)
    std_score = np.std(bootstrap_scores)
    
    print(f"Bootstrap Results ({n_bootstrap} iterations):")
    print(f"Mean Score: {mean_score:.4f}")
    print(f"Std Dev: {std_score:.4f}")
    print(f"{confidence_level*100}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'mean': mean_score,
        'std': std_score,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_scores': bootstrap_scores
    }

# Usage example
result = paired_t_test(preds_v1, preds_v2, labels)
mcnemar_result = mcnemar_test(preds_v1, preds_v2, labels)
ci_result = bootstrap_confidence_interval(preds_v2, labels, accuracy_score)
```

### Multiple Comparison Correction

```python
from statsmodels.stats.multitest import multipletests

def compare_multiple_models(model_predictions_list, ground_truth, method='fdr_bh'):
    """
    Compare multiple models with correction for multiple comparisons.
    
    Args:
        model_predictions_list: List of (model_name, predictions) tuples
        ground_truth: True labels
        method: Correction method ('bonferroni', 'fdr_bh', 'holm', etc.)
    """
    # Use first model as baseline
    baseline_name, baseline_preds = model_predictions_list[0]
    baseline_correct = (baseline_preds == ground_truth).astype(int)
    
    p_values = []
    model_names = []
    
    for model_name, model_preds in model_predictions_list[1:]:
        model_correct = (model_preds == ground_truth).astype(int)
        
        # Paired t-test against baseline
        _, p_value = stats.ttest_rel(model_correct, baseline_correct)
        p_values.append(p_value)
        model_names.append(model_name)
    
    # Apply correction
    reject, corrected_p_values, _, _ = multipletests(
        p_values, 
        alpha=0.05, 
        method=method
    )
    
    print(f"Multiple Comparison Correction: {method}")
    print(f"{'Model':<20} {'Raw P':<10} {'Corrected P':<12} {'Significant'}")
    print("-" * 55)
    
    for name, raw_p, corr_p, sig in zip(model_names, p_values, corrected_p_values, reject):
        print(f"{name:<20} {raw_p:<10.6f} {corr_p:<12.6f} {'Yes' if sig else 'No'}")
    
    return {
        'model_names': model_names,
        'raw_p_values': p_values,
        'corrected_p_values': corrected_p_values,
        'reject_null': reject
    }
```

---

## Bias and Fairness Detection

### Demographic Parity and Equalized Odds

```python
import pandas as pd
from sklearn.metrics import confusion_matrix

class FairnessAuditor:
    def __init__(self, predictions, ground_truth, sensitive_attributes):
        """
        Args:
            predictions: Model predictions
            ground_truth: True labels
            sensitive_attributes: Dict of protected attributes 
                                 (e.g., {'gender': [...], 'race': [...]})
        """
        self.predictions = np.array(predictions)
        self.ground_truth = np.array(ground_truth)
        self.sensitive_attributes = sensitive_attributes
        
    def demographic_parity(self, attribute_name):
        """
        Check if positive prediction rates are equal across groups.
        
        Demographic parity: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
        """
        attribute = self.sensitive_attributes[attribute_name]
        unique_groups = np.unique(attribute)
        
        positive_rates = {}
        for group in unique_groups:
            mask = attribute == group
            positive_rate = np.mean(self.predictions[mask] == 1)
            positive_rates[group] = positive_rate
        
        # Calculate disparity
        rates = list(positive_rates.values())
        max_disparity = max(rates) - min(rates)
        
        print(f"Demographic Parity for '{attribute_name}':")
        print(f"{'Group':<15} {'Positive Rate':<15}")
        print("-" * 30)
        for group, rate in positive_rates.items():
            print(f"{str(group):<15} {rate:.4f}")
        print(f"\nMax Disparity: {max_disparity:.4f}")
        
        # Rule of thumb: disparity < 0.1 is acceptable
        passed = max_disparity < 0.1
        print(f"Status: {'PASS' if passed else 'FAIL'} (threshold: 0.1)")
        
        return {
            'positive_rates': positive_rates,
            'max_disparity': max_disparity,
            'passed': passed
        }
    
    def equalized_odds(self, attribute_name):
        """
        Check if TPR and FPR are equal across groups.
        
        Equalized odds: P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)
                       and P(Ŷ=1|Y=0,A=0) = P(Ŷ=1|Y=0,A=1)
        """
        attribute = self.sensitive_attributes[attribute_name]
        unique_groups = np.unique(attribute)
        
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in unique_groups:
            mask = attribute == group
            
            # True Positive Rate (Recall)
            positive_mask = mask & (self.ground_truth == 1)
            if np.sum(positive_mask) > 0:
                tpr = np.mean(self.predictions[positive_mask] == 1)
            else:
                tpr = 0.0
            
            # False Positive Rate
            negative_mask = mask & (self.ground_truth == 0)
            if np.sum(negative_mask) > 0:
                fpr = np.mean(self.predictions[negative_mask] == 1)
            else:
                fpr = 0.0
            
            tpr_by_group[group] = tpr
            fpr_by_group[group] = fpr
        
        # Calculate disparities
        tpr_disparity = max(tpr_by_group.values()) - min(tpr_by_group.values())
        fpr_disparity = max(fpr_by_group.values()) - min(fpr_by_group.values())
        
        print(f"\nEqualized Odds for '{attribute_name}':")
        print(f"{'Group':<10} {'TPR':<10} {'FPR':<10}")
        print("-" * 30)
        for group in unique_groups:
            print(f"{str(group):<10} {tpr_by_group[group]:.4f}   {fpr_by_group[group]:.4f}")
        
        print(f"\nTPR Disparity: {tpr_disparity:.4f}")
        print(f"FPR Disparity: {fpr_disparity:.4f}")
        
        passed = (tpr_disparity < 0.1) and (fpr_disparity < 0.1)
        print(f"Status: {'PASS' if passed else 'FAIL'} (threshold: 0.1)")
        
        return {
            'tpr_by_group': tpr_by_group,
            'fpr_by_group': fpr_by_group,
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'passed': passed
        }
    
    def predictive_parity(self, attribute_name):
        """
        Check if precision is equal across groups.
        
        Predictive parity: P(Y=1|Ŷ=1,A=0) = P(Y=1|Ŷ=1,A=1)
        """
        attribute = self.sensitive_attributes[attribute_name]
        unique_groups = np.unique(attribute)
        
        precision_by_group = {}
        
        for group in unique_groups:
            mask = attribute == group
            predicted_positive_mask = mask & (self.predictions == 1)
            
            if np.sum(predicted_positive_mask) > 0:
                precision = np.mean(self.ground_truth[predicted_positive_mask] == 1)
            else:
                precision = 0.0
            
            precision_by_group[group] = precision
        
        disparity = max(precision_by_group.values()) - min(precision_by_group.values())
        
        print(f"\nPredictive Parity for '{attribute_name}':")
        print(f"{'Group':<15} {'Precision':<15}")
        print("-" * 30)
        for group, prec in precision_by_group.items():
            print(f"{str(group):<15} {prec:.4f}")
        print(f"\nDisparity: {disparity:.4f}")
        
        passed = disparity < 0.1
        print(f"Status: {'PASS' if passed else 'FAIL'}")
        
        return {
            'precision_by_group': precision_by_group,
            'disparity': disparity,
            'passed': passed
        }
    
    def generate_fairness_report(self):
        """Generate comprehensive fairness report for all attributes."""
        report = {}
        
        for attr_name in self.sensitive_attributes.keys():
            print(f"\n{'='*60}")
            print(f"FAIRNESS AUDIT: {attr_name.upper()}")
            print(f"{'='*60}")
            
            report[attr_name] = {
                'demographic_parity': self.demographic_parity(attr_name),
                'equalized_odds': self.equalized_odds(attr_name),
                'predictive_parity': self.predictive_parity(attr_name)
            }
        
        return report

# Usage example
auditor = FairnessAuditor(
    predictions=model_preds,
    ground_truth=true_labels,
    sensitive_attributes={
        'gender': gender_array,
        'age_group': age_group_array,
        'region': region_array
    }
)

fairness_report = auditor.generate_fairness_report()
```

### Bias Mitigation Strategies

```python
class BiasMitigator:
    def __init__(self, model, training_data, sensitive_attributes):
        self.model = model
        self.training_data = training_data
        self.sensitive_attributes = sensitive_attributes
    
    def reweighting(self, target_attribute):
        """
        Reweight samples to balance representation across groups.
        """
        attribute = self.sensitive_attributes[target_attribute]
        unique_groups, counts = np.unique(attribute, return_counts=True)
        
        # Calculate weights inversely proportional to group size
        total_samples = len(attribute)
        weights = np.zeros_like(attribute, dtype=float)
        
        for group, count in zip(unique_groups, counts):
            mask = attribute == group
            # Weight = total_samples / (num_groups * count_in_group)
            weights[mask] = total_samples / (len(unique_groups) * count)
        
        # Normalize weights
        weights = weights * len(attribute) / np.sum(weights)
        
        print(f"Reweighting for '{target_attribute}':")
        print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        
        return weights
    
    def adversarial_debiasing(self, debias_epochs=10):
        """
        Train adversary to predict sensitive attribute from representations.
        Update model to minimize adversary's success.
        """
        # Implementation would add adversarial head to model
        # and alternate between main task and adversarial training
        pass
    
    def threshold_optimization(self, val_predictions, val_labels, 
                               sensitive_attributes, target_attribute):
        """
        Optimize decision thresholds per group to equalize metrics.
        """
        attribute = sensitive_attributes[target_attribute]
        unique_groups = np.unique(attribute)
        
        optimal_thresholds = {}
        
        for group in unique_groups:
            mask = attribute == group
            group_preds = val_predictions[mask]
            group_labels = val_labels[mask]
            
            # Find threshold that equalizes TPR across groups
            best_threshold = 0.5
            best_metric = 0
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                binarized_preds = (group_preds > threshold).astype(int)
                tpr = np.sum((binarized_preds == 1) & (group_labels == 1)) / \
                      np.sum(group_labels == 1)
                
                # Optimize for equal TPR (simplified)
                if tpr > best_metric:
                    best_metric = tpr
                    best_threshold = threshold
            
            optimal_thresholds[group] = best_threshold
        
        print(f"Optimal thresholds for '{target_attribute}':")
        for group, thresh in optimal_thresholds.items():
            print(f"  Group {group}: {thresh:.2f}")
        
        return optimal_thresholds
```

---

## Robustness Testing

### Perturbation Testing

```python
import random
import string

class RobustnessTester:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def character_level_perturbations(self, texts, perturbation_rate=0.1):
        """
        Test robustness to character-level noise.
        
        Types:
        - Random character insertion
        - Random character deletion
        - Random character substitution
        - Adjacent character swap
        """
        perturbed_texts = []
        
        for text in texts:
            chars = list(text)
            num_perturbations = max(1, int(len(chars) * perturbation_rate))
            
            for _ in range(num_perturbations):
                op = random.choice(['insert', 'delete', 'substitute', 'swap'])
                idx = random.randint(0, len(chars) - 1)
                
                if op == 'insert' and chars[idx].isalpha():
                    chars.insert(idx, random.choice(string.ascii_lowercase))
                elif op == 'delete' and len(chars) > 1:
                    chars.pop(idx)
                elif op == 'substitute' and chars[idx].isalpha():
                    chars[idx] = random.choice(string.ascii_lowercase)
                elif op == 'swap' and idx < len(chars) - 1:
                    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            
            perturbed_texts.append(''.join(chars))
        
        return perturbed_texts
    
    def word_level_perturbations(self, texts, perturbation_rate=0.1):
        """
        Test robustness to word-level noise.
        
        Types:
        - Random word deletion
        - Random word insertion (synonyms or random)
        - Word order shuffling (local)
        """
        perturbed_texts = []
        
        for text in texts:
            words = text.split()
            num_perturbations = max(1, int(len(words) * perturbation_rate))
            
            for _ in range(num_perturbations):
                op = random.choice(['delete', 'insert', 'shuffle'])
                idx = random.randint(0, len(words) - 1)
                
                if op == 'delete' and len(words) > 1:
                    words.pop(idx)
                elif op == 'insert':
                    # Insert random common word
                    common_words = ['the', 'a', 'is', 'it', 'this', 'that']
                    words.insert(idx, random.choice(common_words))
                elif op == 'shuffle' and idx < len(words) - 1:
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]
            
            perturbed_texts.append(' '.join(words))
        
        return perturbed_texts
    
    def synonym_replacement(self, texts, replacement_rate=0.1):
        """
        Replace words with synonyms using WordNet or similar.
        """
        try:
            from nltk.corpus import wordnet
            from nltk import word_tokenize, pos_tag
        except ImportError:
            print("NLTK not available. Install with: pip install nltk")
            return texts
        
        perturbed_texts = []
        
        for text in texts:
            words = text.split()
            num_replacements = max(1, int(len(words) * replacement_rate))
            
            for _ in range(num_replacements):
                idx = random.randint(0, len(words) - 1)
                word = words[idx]
                
                # Get synonyms
                synsets = wordnet.synsets(word)
                if synsets:
                    synonyms = []
                    for synset in synsets:
                        for lemma in synset.lemmas():
                            synonym = lemma.name().replace('_', ' ')
                            if synonym.lower() != word.lower():
                                synonyms.append(synonym)
                    
                    if synonyms:
                        words[idx] = random.choice(synonyms)
            
            perturbed_texts.append(' '.join(words))
        
        return perturbed_texts
    
    def back_translation(self, texts, intermediate_lang='de'):
        """
        Test robustness via back-translation (round-trip translation).
        
        Translate to intermediate language and back.
        """
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # Load translation models
            trans_to_tokenizer = MarianTokenizer.from_pretrained(
                f'Helsinki-NLP/opus-mt-en-{intermediate_lang}'
            )
            trans_to_model = MarianMTModel.from_pretrained(
                f'Helsinki-NLP/opus-mt-en-{intermediate_lang}'
            )
            
            trans_back_tokenizer = MarianTokenizer.from_pretrained(
                f'Helsinki-NLP/opus-mt-{intermediate_lang}-en'
            )
            trans_back_model = MarianMTModel.from_pretrained(
                f'Helsinki-NLP/opus-mt-{intermediate_lang}-en'
            )
        except Exception as e:
            print(f"Translation models not available: {e}")
            return texts
        
        perturbed_texts = []
        
        for text in texts:
            # Translate to intermediate language
            inputs_to = trans_to_tokenizer(text, return_tensors='pt', padding=True)
            translated_to = trans_to_model.generate(**inputs_to)
            intermediate = trans_to_tokenizer.decode(translated_to[0], skip_special_tokens=True)
            
            # Translate back to English
            inputs_back = trans_back_tokenizer(intermediate, return_tensors='pt', padding=True)
            translated_back = trans_back_model.generate(**inputs_back)
            back_translated = trans_back_tokenizer.decode(
                translated_back[0], 
                skip_special_tokens=True
            )
            
            perturbed_texts.append(back_translated)
        
        return perturbed_texts
    
    def evaluate_robustness(self, original_texts, labels, perturbation_fn, 
                           perturbation_name):
        """
        Evaluate model performance on perturbed data.
        """
        perturbed_texts = perturbation_fn(original_texts)
        
        # Get predictions for original and perturbed
        orig_preds = self.model.predict(original_texts)
        pert_preds = self.model.predict(perturbed_texts)
        
        # Calculate metrics
        orig_accuracy = np.mean(orig_preds == labels)
        pert_accuracy = np.mean(pert_preds == labels)
        
        # Consistency: predictions unchanged despite perturbation
        consistency = np.mean(orig_preds == pert_preds)
        
        print(f"\nRobustness Test: {perturbation_name}")
        print(f"Original Accuracy: {orig_accuracy:.4f}")
        print(f"Perturbed Accuracy: {pert_accuracy:.4f}")
        print(f"Accuracy Drop: {orig_accuracy - pert_accuracy:.4f}")
        print(f"Prediction Consistency: {consistency:.4f}")
        
        return {
            'original_accuracy': orig_accuracy,
            'perturbed_accuracy': pert_accuracy,
            'accuracy_drop': orig_accuracy - pert_accuracy,
            'consistency': consistency
        }
    
    def comprehensive_robustness_suite(self, texts, labels):
        """Run all robustness tests."""
        results = {}
        
        tests = [
            ('Character Insertion', lambda t: self.character_level_perturbations(t, 0.1)),
            ('Character Deletion', lambda t: self.character_level_perturbations(t, 0.1)),
            ('Word Deletion', lambda t: self.word_level_perturbations(t, 0.1)),
            ('Word Insertion', lambda t: self.word_level_perturbations(t, 0.1)),
            ('Synonym Replacement', lambda t: self.synonym_replacement(t, 0.1)),
        ]
        
        for test_name, pert_fn in tests:
            results[test_name] = self.evaluate_robustness(
                texts, labels, pert_fn, test_name
            )
        
        # Summary
        print(f"\n{'='*60}")
        print("ROBUSTNESS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Test':<25} {'Acc Drop':<12} {'Consistency':<12}")
        print("-" * 50)
        
        for test_name, metrics in results.items():
            print(f"{test_name:<25} {metrics['accuracy_drop']:.4f}      {metrics['consistency']:.4f}")
        
        return results
```

### Stress Testing

```python
class StressTester:
    def __init__(self, model):
        self.model = model
    
    def length_stress_test(self, texts, labels):
        """Test performance across different input lengths."""
        lengths = [len(text.split()) for text in texts]
        
        # Bin by length
        bins = [(0, 10), (10, 25), (25, 50), (50, 100), (100, float('inf'))]
        results = {}
        
        for min_len, max_len in bins:
            mask = [(min_len <= l < max_len) for l in lengths]
            bin_texts = [t for t, m in zip(texts, mask) if m]
            bin_labels = [l for l, m in zip(labels, mask) if m]
            
            if len(bin_labels) > 0:
                preds = self.model.predict(bin_texts)
                accuracy = np.mean(preds == bin_labels)
                
                bin_name = f"{min_len}-{max_len if max_len != float('inf') else '∞'}"
                results[bin_name] = {
                    'count': len(bin_labels),
                    'accuracy': accuracy
                }
                
                print(f"Length {bin_name}: {accuracy:.4f} (n={len(bin_labels)})")
        
        return results
    
    def rare_class_stress_test(self, texts, labels):
        """Test performance on rare/underrepresented classes."""
        unique_classes, counts = np.unique(labels, return_counts=True)
        
        results = {}
        
        for cls, count in zip(unique_classes, counts):
            mask = labels == cls
            cls_texts = [t for t, m in zip(texts, mask) if m]
            cls_labels = [l for l, m in zip(labels, mask) if m]
            
            preds = self.model.predict(cls_texts)
            accuracy = np.mean(preds == cls_labels)
            
            frequency = 'rare' if count < len(labels) * 0.05 else 'common'
            
            results[cls] = {
                'count': count,
                'frequency': frequency,
                'accuracy': accuracy
            }
            
            print(f"Class {cls}: {accuracy:.4f} (n={count}, {frequency})")
        
        return results
    
    def edge_case_stress_test(self, edge_cases):
        """Test specific edge cases."""
        results = {}
        
        for case_name, (text, expected_label) in edge_cases.items():
            pred = self.model.predict([text])[0]
            correct = pred == expected_label
            
            results[case_name] = {
                'expected': expected_label,
                'predicted': pred,
                'correct': correct
            }
            
            status = "✓" if correct else "✗"
            print(f"{status} {case_name}: Expected={expected_label}, Got={pred}")
        
        return results
```

---

## Adversarial Testing

### TextFooler-Style Adversarial Attacks

```python
class AdversarialAttacker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def calculate_importance_scores(self, text, label):
        """
        Calculate importance score for each word by masking.
        """
        words = text.split()
        importance_scores = []
        
        # Get original prediction probability
        orig_probs = self.model.predict_proba([text])[0]
        orig_prob = orig_probs[label]
        
        for i, word in enumerate(words):
            # Mask this word
            masked_words = words[:i] + ['[MASK]'] + words[i+1:]
            masked_text = ' '.join(masked_words)
            
            # Get prediction with masked word
            masked_probs = self.model.predict_proba([masked_text])[0]
            masked_prob = masked_probs[label]
            
            # Importance = drop in probability
            importance = orig_prob - masked_prob
            importance_scores.append(importance)
        
        return importance_scores
    
    def find_synonyms(self, word, top_k=10):
        """Find synonyms for a word."""
        try:
            from nltk.corpus import wordnet
        except ImportError:
            return [word]
        
        synonyms = []
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and synonym.isalpha():
                    synonyms.append(synonym)
        
        return synonyms[:top_k]
    
    def textfooler_attack(self, text, true_label, max_iterations=10):
        """
        Implement TextFooler-style adversarial attack.
        
        Strategy:
        1. Identify important words
        2. Replace with synonyms that change prediction
        3. Ensure semantic similarity and grammaticality
        """
        words = text.split()
        current_text = text
        attacked_words = set()
        
        for iteration in range(max_iterations):
            # Get current prediction
            pred = self.model.predict([current_text])[0]
            
            # Check if attack succeeded
            if pred != true_label:
                print(f"Attack succeeded at iteration {iteration + 1}")
                return {
                    'success': True,
                    'adversarial_text': current_text,
                    'iterations': iteration + 1,
                    'attacked_words': attacked_words
                }
            
            # Calculate importance scores
            importance_scores = self.calculate_importance_scores(
                current_text, true_label
            )
            
            # Sort words by importance (descending)
            sorted_indices = np.argsort(importance_scores)[::-1]
            
            # Try to replace most important unattacked word
            attacked = False
            for idx in sorted_indices:
                if idx in attacked_words:
                    continue
                
                word = words[idx]
                synonyms = self.find_synonyms(word)
                
                # Try each synonym
                for synonym in synonyms:
                    # Create candidate text
                    candidate_words = words.copy()
                    candidate_words[idx] = synonym
                    candidate_text = ' '.join(candidate_words)
                    
                    # Check if prediction changes
                    new_pred = self.model.predict([candidate_text])[0]
                    
                    if new_pred != true_label:
                        # Attack successful
                        current_text = candidate_text
                        words = candidate_words
                        attacked_words.add(idx)
                        attacked = True
                        break
                    
                    # Also check if probability of true label decreases
                    orig_probs = self.model.predict_proba([current_text])[0]
                    new_probs = self.model.predict_proba([candidate_text])[0]
                    
                    if new_probs[true_label] < orig_probs[true_label]:
                        # Accept if it reduces confidence
                        current_text = candidate_text
                        words = candidate_words
                        attacked_words.add(idx)
                        attacked = True
                        break
                
                if attacked:
                    break
            
            if not attacked:
                print("No successful perturbation found")
                break
        
        return {
            'success': False,
            'adversarial_text': current_text,
            'iterations': max_iterations,
            'attacked_words': attacked_words
        }
    
    def generate_adversarial_dataset(self, texts, labels, attack_rate=0.3):
        """
        Generate adversarial examples for a portion of the dataset.
        """
        num_to_attack = int(len(texts) * attack_rate)
        indices = np.random.choice(len(texts), num_to_attack, replace=False)
        
        adversarial_examples = []
        success_count = 0
        
        for idx in indices:
            text = texts[idx]
            label = labels[idx]
            
            result = self.textfooler_attack(text, label)
            
            if result['success']:
                success_count += 1
                adversarial_examples.append({
                    'original': text,
                    'adversarial': result['adversarial_text'],
                    'label': label,
                    'iterations': result['iterations']
                })
        
        attack_success_rate = success_count / num_to_attack
        
        print(f"\nAdversarial Attack Summary:")
        print(f"Attempted: {num_to_attack}")
        print(f"Successful: {success_count}")
        print(f"Success Rate: {attack_success_rate:.4f}")
        
        return adversarial_examples, attack_success_rate
```

### Adversarial Training

```python
class AdversarialTrainer:
    def __init__(self, model, attacker, config):
        self.model = model
        self.attacker = attacker
        self.config = config
    
    def train_with_adversarial_examples(self, train_loader, adversarial_ratio=0.5):
        """
        Train model augmented with adversarial examples.
        
        Mix of clean and adversarial examples improves robustness.
        """
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            
            for batch_idx, (texts, labels) in enumerate(train_loader):
                # Get adversarial examples for this batch
                adv_examples = []
                adv_labels = []
                
                for text, label in zip(texts, labels):
                    if random.random() < adversarial_ratio:
                        result = self.attacker.textfooler_attack(text, label, max_iterations=5)
                        if result['success']:
                            adv_examples.append(result['adversarial_text'])
                            adv_labels.append(label)
                        else:
                            adv_examples.append(text)
                            adv_labels.append(label)
                    else:
                        adv_examples.append(text)
                        adv_labels.append(label)
                
                # Train on mixed batch
                loss = self.model.train_step(adv_examples, adv_labels)
                total_loss += loss
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        return self.model
```

---

## Domain Shift Detection

### Distribution Comparison

```python
from scipy.spatial.distance import jensenshannon, wasserstein_distance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class DomainShiftDetector:
    def __init__(self, source_data, target_data, model):
        """
        Args:
            source_data: Training distribution data
            target_data: New/target distribution data
            model: Trained model for extracting representations
        """
        self.source_data = source_data
        self.target_data = target_data
        self.model = model
    
    def extract_representations(self, texts):
        """Extract hidden representations from model."""
        representations = []
        
        for text in texts:
            rep = self.model.get_hidden_representation(text)
            representations.append(rep)
        
        return np.array(representations)
    
    def compare_label_distributions(self, source_labels, target_labels):
        """Compare label distribution between domains."""
        # Get unique labels
        all_labels = np.unique(np.concatenate([source_labels, target_labels]))
        
        # Calculate distributions
        source_dist = np.array([np.mean(source_labels == l) for l in all_labels])
        target_dist = np.array([np.mean(target_labels == l) for l in all_labels])
        
        # Jensen-Shannon divergence
        js_div = jensenshannon(source_dist, target_dist)
        
        # Total Variation Distance
        tv_dist = 0.5 * np.sum(np.abs(source_dist - target_dist))
        
        print(f"Label Distribution Comparison:")
        print(f"Jensen-Shannon Divergence: {js_div:.4f}")
        print(f"Total Variation Distance: {tv_dist:.4f}")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(all_labels))
        width = 0.35
        
        ax.bar(x - width/2, source_dist, width, label='Source', alpha=0.8)
        ax.bar(x + width/2, target_dist, width, label='Target', alpha=0.08)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Proportion')
        ax.set_title('Label Distribution: Source vs Target')
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('label_distribution_comparison.png')
        plt.show()
        
        return {
            'js_divergence': js_div,
            'tv_distance': tv_dist,
            'source_distribution': source_dist,
            'target_distribution': target_dist
        }
    
    def compare_feature_distributions(self):
        """Compare feature distributions using statistical tests."""
        # Extract representations
        print("Extracting source representations...")
        source_reps = self.extract_representations(self.source_data)
        
        print("Extracting target representations...")
        target_reps = self.extract_representations(self.target_data)
        
        # Per-feature comparison (Kolmogorov-Smirnov test)
        n_features = source_reps.shape[1]
        ks_statistics = []
        ks_p_values = []
        
        for i in range(min(n_features, 100)):  # Sample features for efficiency
            stat, p_val = stats.ks_2samp(source_reps[:, i], target_reps[:, i])
            ks_statistics.append(stat)
            ks_p_values.append(p_val)
        
        # Summary statistics
        mean_ks_stat = np.mean(ks_statistics)
        frac_significant = np.mean([p < 0.05 for p in ks_p_values])
        
        print(f"\nFeature Distribution Comparison:")
        print(f"Mean KS Statistic: {mean_ks_stat:.4f}")
        print(f"Fraction of Significant Features (p<0.05): {frac_significant:.4f}")
        
        # Wasserstein distance on aggregated representations
        source_means = np.mean(source_reps, axis=0)
        target_means = np.mean(target_reps, axis=0)
        
        wasserstein_dist = wasserstein_distance(source_means, target_means)
        print(f"Wasserstein Distance (means): {wasserstein_dist:.4f}")
        
        return {
            'mean_ks_statistic': mean_ks_stat,
            'frac_significant': frac_significant,
            'wasserstein_distance': wasserstein_dist
        }
    
    def visualize_domain_shift(self):
        """Visualize domain shift using t-SNE."""
        # Combine data
        all_data = np.concatenate([self.source_data, self.target_data])
        all_labels = ['Source'] * len(self.source_data) + \
                     ['Target'] * len(self.target_data)
        
        # Extract representations
        reps = self.extract_representations(all_data)
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reps_2d = tsne.fit_transform(reps)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        source_mask = np.array(all_labels) == 'Source'
        target_mask = np.array(all_labels) == 'Target'
        
        ax.scatter(reps_2d[source_mask, 0], reps_2d[source_mask, 1],
                  alpha=0.5, label='Source', s=10)
        ax.scatter(reps_2d[target_mask, 0], reps_2d[target_mask, 1],
                  alpha=0.5, label='Target', s=10)
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('Domain Shift Visualization: Source vs Target')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('domain_shift_tsne.png')
        plt.show()
    
    def detect_covariate_shift(self):
        """Detect covariate shift using KLIEP-like method."""
        # Simplified covariate shift detection
        source_reps = self.extract_representations(self.source_data)
        target_reps = self.extract_representations(self.target_data)
        
        # Train classifier to distinguish source from target
        X = np.concatenate([source_reps, target_reps])
        y = np.concatenate([
            np.zeros(len(source_reps)),
            np.ones(len(target_reps))
        ])
        
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        
        # If classifier can easily distinguish, there's significant shift
        accuracy = clf.score(X, y)
        
        print(f"Covariate Shift Detection:")
        print(f"Source/Target Classifier Accuracy: {accuracy:.4f}")
        
        if accuracy > 0.7:
            print("WARNING: Significant covariate shift detected!")
        elif accuracy > 0.55:
            print("MODERATE: Some covariate shift present")
        else:
            print("LOW: Minimal covariate shift")
        
        return {
            'classifier_accuracy': accuracy,
            'shift_severity': 'high' if accuracy > 0.7 else 
                             'moderate' if accuracy > 0.55 else 'low'
        }
```

---

## Calibration and Confidence Estimation

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

class CalibrationAnalyzer:
    def __init__(self, model):
        self.model = model
    
    def get_predicted_probabilities(self, texts):
        """Get predicted probabilities from model."""
        return self.model.predict_proba(texts)
    
    def plot_calibration_curve(self, texts, labels, n_bins=10):
        """
        Plot reliability diagram (calibration curve).
        """
        probs = self.get_predicted_probabilities(texts)
        
        # For binary classification
        if probs.shape[1] == 2:
            prob_positive = probs[:, 1]
        else:
            # Use max probability for multiclass
            prob_positive = np.max(probs, axis=1)
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, prob_positive, n_bins=n_bins
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
               label='Model', markersize=10)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve (Reliability Diagram)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('calibration_curve.png')
        plt.show()
        
        # Calculate Expected Calibration Error (ECE)
        ece = self.calculate_ece(labels, prob_positive, n_bins)
        
        return {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'ece': ece
        }
    
    def calculate_ece(self, labels, probabilities, n_bins=10):
        """
        Calculate Expected Calibration Error.
        
        ECE measures the weighted average difference between predicted 
        confidence and actual accuracy across bins.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            # Find samples in this bin
            in_bin = (probabilities > bin_boundaries[i]) & \
                     (probabilities <= bin_boundaries[i + 1])
            
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Average confidence in bin
                avg_confidence = np.mean(probabilities[in_bin])
                
                # Actual accuracy in bin
                avg_accuracy = np.mean(labels[in_bin] == 
                                      (probabilities[in_bin] > 0.5).astype(int))
                
                # Weighted difference
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        
        return ece
    
    def temperature_scaling(self, val_texts, val_labels):
        """
        Apply temperature scaling to improve calibration.
        
        Find optimal temperature T that minimizes NLL on validation set.
        """
        from scipy.optimize import minimize_scalar
        
        logits = self.model.get_logits(val_texts)
        
        def nll_loss(T):
            scaled_logits = logits / T
            probs = softmax(scaled_logits, axis=1)
            
            # Negative log likelihood
            nll = -np.mean(np.log(probs[np.arange(len(val_labels)), val_labels]))
            return nll
        
        # Find optimal temperature
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        optimal_T = result.x
        
        print(f"Optimal Temperature: {optimal_T:.4f}")
        print(f"NLL before scaling: {nll_loss(1.0):.4f}")
        print(f"NLL after scaling: {nll_loss(optimal_T):.4f}")
        
        return optimal_T
    
    def apply_temperature(self, texts, temperature):
        """Apply temperature scaling to predictions."""
        logits = self.model.get_logits(texts)
        scaled_logits = logits / temperature
        calibrated_probs = softmax(scaled_logits, axis=1)
        
        return calibrated_probs
```

---

## A/B Testing Framework

```python
import pandas as pd
from datetime import datetime, timedelta

class ABTestingFramework:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.results = []
    
    def design_experiment(self, control_model, treatment_model, 
                         sample_size, duration_days, metrics):
        """
        Design A/B test with proper power analysis.
        """
        from statsmodels.stats.power import zt_ind_solve_power
        
        # Power analysis
        effect_size = 0.1  # Minimum detectable effect (10%)
        alpha = 0.05
        power = 0.8
        
        required_n = zt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1  # Equal split
        )
        
        required_n = int(np.ceil(required_n))
        
        print(f"A/B Test Design: {self.experiment_name}")
        print(f"Required samples per variant: {required_n}")
        print(f"Total required: {required_n * 2}")
        print(f"Planned sample size: {sample_size}")
        print(f"Duration: {duration_days} days")
        
        if sample_size < required_n:
            print(f"WARNING: Planned sample size may be underpowered!")
        
        return {
            'required_per_variant': required_n,
            'planned_per_variant': sample_size // 2,
            'effect_size': effect_size,
            'alpha': alpha,
            'power': power
        }
    
    def assign_users(self, user_ids, assignment_ratio=0.5):
        """
        Randomly assign users to control or treatment.
        """
        np.random.seed(42)  # Reproducible assignment
        
        assignments = np.random.rand(len(user_ids)) < assignment_ratio
        user_assignments = {
            uid: 'treatment' if assign else 'control'
            for uid, assign in zip(user_ids, assignments)
        }
        
        # Verify balance
        n_control = sum(1 for v in user_assignments.values() if v == 'control')
        n_treatment = len(user_ids) - n_control
        
        print(f"User Assignment:")
        print(f"Control: {n_control} ({n_control/len(user_ids)*100:.1f}%)")
        print(f"Treatment: {n_treatment} ({n_treatment/len(user_ids)*100:.1f}%)")
        
        return user_assignments
    
    def collect_metrics(self, interactions, user_assignments):
        """
        Collect and aggregate metrics from user interactions.
        """
        # Add assignment to interactions
        df = pd.DataFrame(interactions)
        df['variant'] = df['user_id'].map(user_assignments)
        
        # Aggregate by variant
        results = {}
        
        for variant in ['control', 'treatment']:
            variant_data = df[df['variant'] == variant]
            
            variant_results = {
                'n_users': variant_data['user_id'].nunique(),
                'n_interactions': len(variant_data),
            }
            
            # Calculate each metric
            for metric in ['accuracy', 'latency', 'user_satisfaction']:
                if metric in variant_data.columns:
                    variant_results[f'{metric}_mean'] = variant_data[metric].mean()
                    variant_results[f'{metric}_std'] = variant_data[metric].std()
            
            results[variant] = variant_results
        
        return results
    
    def analyze_results(self, control_data, treatment_data, metric_name):
        """
        Analyze A/B test results with statistical testing.
        """
        control_values = np.array(control_data)
        treatment_values = np.array(treatment_data)
        
        # Difference in means
        diff = np.mean(treatment_values) - np.mean(control_values)
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
        
        # Confidence interval for difference
        pooled_se = np.sqrt(np.var(control_values)/len(control_values) + 
                           np.var(treatment_values)/len(treatment_values))
        ci_lower = diff - 1.96 * pooled_se
        ci_upper = diff + 1.96 * pooled_se
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(control_values) + np.var(treatment_values)) / 2)
        cohens_d = diff / pooled_std
        
        print(f"\nA/B Test Analysis: {metric_name}")
        print(f"Control Mean: {np.mean(control_values):.4f}")
        print(f"Treatment Mean: {np.mean(treatment_values):.4f}")
        print(f"Difference: {diff:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Cohen's d: {cohens_d:.4f}")
        
        # Interpretation
        if p_value < 0.05:
            direction = "better" if diff > 0 else "worse"
            print(f"Result: Treatment is statistically significantly {direction}")
            
            if abs(cohens_d) < 0.2:
                print("Effect size: Small")
            elif abs(cohens_d) < 0.5:
                print("Effect size: Medium")
            else:
                print("Effect size: Large")
        else:
            print("Result: No statistically significant difference")
        
        return {
            'difference': diff,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
    
    def sequential_testing(self, daily_results, stopping_rule='pocket'):
        """
        Sequential A/B testing with early stopping.
        
        Allows monitoring and early stopping if results are clear.
        """
        cumulative_control = []
        cumulative_treatment = []
        
        decisions = []
        
        for day, day_data in enumerate(daily_results):
            cumulative_control.extend(day_data['control'])
            cumulative_treatment.extend(day_data['treatment'])
            
            # Daily analysis
            if len(cumulative_control) > 100 and len(cumulative_treatment) > 100:
                result = self.analyze_results(
                    cumulative_control, 
                    cumulative_treatment,
                    'primary_metric'
                )
                
                # Stopping rule
                if result['p_value'] < 0.01:  # Strong evidence
                    decision = 'STOP_EARLY'
                elif day >= len(daily_results) - 1:
                    decision = 'CONCLUDE'
                else:
                    decision = 'CONTINUE'
                
                decisions.append({
                    'day': day,
                    'decision': decision,
                    'p_value': result['p_value']
                })
                
                print(f"Day {day}: p={result['p_value']:.6f}, Decision: {decision}")
                
                if decision == 'STOP_EARLY':
                    print("Early stopping triggered!")
                    break
        
        return decisions
```

---

## Regression Testing for Models

```python
import json
import hashlib

class ModelRegressionTester:
    def __init__(self, baseline_model_path, test_suite_path):
        """
        Initialize regression testing framework.
        """
        self.baseline_model = self.load_model(baseline_model_path)
        self.test_suite = self.load_test_suite(test_suite_path)
        self.baseline_results = self.run_baseline()
    
    def load_test_suite(self, path):
        """Load or create test suite."""
        try:
            with open(path, 'r') as f:
                test_suite = json.load(f)
        except FileNotFoundError:
            # Create default test suite
            test_suite = {
                'functional_tests': [],
                'edge_cases': [],
                'performance_tests': [],
                'bias_tests': []
            }
        
        return test_suite
    
    def add_functional_test(self, name, input_text, expected_output, 
                           tolerance=0.0):
        """Add a functional test case."""
        test_case = {
            'name': name,
            'input': input_text,
            'expected': expected_output,
            'tolerance': tolerance,
            'type': 'functional'
        }
        
        self.test_suite['functional_tests'].append(test_case)
        self.save_test_suite()
    
    def add_edge_case(self, name, input_text, expected_behavior):
        """Add an edge case test."""
        test_case = {
            'name': name,
            'input': input_text,
            'expected_behavior': expected_behavior,
            'type': 'edge_case'
        }
        
        self.test_suite['edge_cases'].append(test_case)
        self.save_test_suite()
    
    def save_test_suite(self):
        """Save test suite to file."""
        with open('model_test_suite.json', 'w') as f:
            json.dump(self.test_suite, f, indent=2)
    
    def run_baseline(self):
        """Run test suite on baseline model."""
        results = {}
        
        # Functional tests
        functional_results = []
        for test in self.test_suite['functional_tests']:
            prediction = self.baseline_model.predict([test['input']])[0]
            expected = test['expected']
            
            # Check if within tolerance
            if isinstance(expected, (int, float)):
                passed = abs(prediction - expected) <= test['tolerance']
            else:
                passed = prediction == expected
            
            functional_results.append({
                'name': test['name'],
                'passed': passed,
                'prediction': prediction,
                'expected': expected
            })
        
        results['functional'] = functional_results
        
        # Edge cases
        edge_results = []
        for test in self.test_suite['edge_cases']:
            try:
                prediction = self.baseline_model.predict([test['input']])[0]
                behavior = self.check_expected_behavior(prediction, test['expected_behavior'])
                
                edge_results.append({
                    'name': test['name'],
                    'passed': behavior,
                    'prediction': prediction
                })
            except Exception as e:
                edge_results.append({
                    'name': test['name'],
                    'passed': False,
                    'error': str(e)
                })
        
        results['edge_cases'] = edge_results
        
        # Save baseline results
        with open('baseline_regression_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def check_expected_behavior(self, prediction, expected_behavior):
        """Check if prediction matches expected behavior."""
        if expected_behavior == 'high_confidence':
            return prediction['confidence'] > 0.9
        elif expected_behavior == 'uncertain':
            return prediction['confidence'] < 0.6
        elif expected_behavior == 'specific_class':
            return prediction['class'] == expected_behavior['class']
        else:
            return False
    
    def run_regression_test(self, new_model_path):
        """
        Run regression tests on new model and compare to baseline.
        """
        new_model = self.load_model(new_model_path)
        
        regressions = []
        improvements = []
        
        # Compare functional tests
        for baseline_result in self.baseline_results['functional']:
            test_name = baseline_result['name']
            
            # Find corresponding test
            test = next(t for t in self.test_suite['functional_tests'] 
                       if t['name'] == test_name)
            
            # Run on new model
            new_prediction = new_model.predict([test['input']])[0]
            expected = test['expected']
            
            if isinstance(expected, (int, float)):
                new_passed = abs(new_prediction - expected) <= test['tolerance']
            else:
                new_passed = new_prediction == expected
            
            # Compare
            if baseline_result['passed'] and not new_passed:
                regressions.append({
                    'test': test_name,
                    'type': 'functional',
                    'baseline': baseline_result['prediction'],
                    'new': new_prediction,
                    'expected': expected
                })
            elif not baseline_result['passed'] and new_passed:
                improvements.append({
                    'test': test_name,
                    'type': 'functional',
                    'baseline': baseline_result['prediction'],
                    'new': new_prediction
                })
        
        # Report
        print(f"\n{'='*60}")
        print("REGRESSION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Tests: {len(self.baseline_results['functional'])}")
        print(f"Regressions: {len(regressions)}")
        print(f"Improvements: {len(improvements)}")
        
        if regressions:
            print(f"\n⚠️  REGRESSIONS DETECTED:")
            for reg in regressions:
                print(f"  - {reg['test']}: {reg['baseline']} → {reg['new']}")
                print(f"    Expected: {reg['expected']}")
        
        if improvements:
            print(f"\n✅ IMPROVEMENTS:")
            for imp in improvements:
                print(f"  - {imp['test']}: {imp['baseline']} → {imp['new']}")
        
        return {
            'regressions': regressions,
            'improvements': improvements,
            'passed': len(regressions) == 0
        }
    
    def generate_regression_report(self, results):
        """Generate detailed regression report."""
        report = f"""
# Model Regression Test Report

## Summary
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Tests**: {len(self.baseline_results['functional']) + len(self.baseline_results['edge_cases'])}
- **Regressions**: {len(results['regressions'])}
- **Improvements**: {len(results['improvements'])}
- **Status**: {'✅ PASS' if results['passed'] else '❌ FAIL'}

## Regressions
"""
        
        for reg in results['regressions']:
            report += f"""
### {reg['test']}
- Type: {reg['type']}
- Baseline: {reg['baseline']}
- New: {reg['new']}
- Expected: {reg['expected']}
"""
        
        return report
```

---

## Quality Gates and Release Criteria

```python
class QualityGateChecker:
    def __init__(self, release_config):
        """
        Initialize quality gate checker with release criteria.
        
        release_config example:
        {
            'min_accuracy': 0.85,
            'max_fairness_disparity': 0.1,
            'min_robustness_consistency': 0.9,
            'max_calibration_ece': 0.05,
            'no_regressions': True,
            'min_test_coverage': 0.95
        }
        """
        self.config = release_config
    
    def check_all_gates(self, model_metrics):
        """
        Check all quality gates for release readiness.
        """
        gate_results = {}
        all_passed = True
        
        # Accuracy gate
        if 'min_accuracy' in self.config:
            passed = model_metrics['accuracy'] >= self.config['min_accuracy']
            gate_results['accuracy'] = {
                'passed': passed,
                'value': model_metrics['accuracy'],
                'threshold': self.config['min_accuracy']
            }
            if not passed:
                all_passed = False
        
        # Fairness gate
        if 'max_fairness_disparity' in self.config:
            passed = model_metrics['fairness_disparity'] <= self.config['max_fairness_disparity']
            gate_results['fairness'] = {
                'passed': passed,
                'value': model_metrics['fairness_disparity'],
                'threshold': self.config['max_fairness_disparity']
            }
            if not passed:
                all_passed = False
        
        # Robustness gate
        if 'min_robustness_consistency' in self.config:
            passed = model_metrics['robustness_consistency'] >= self.config['min_robustness_consistency']
            gate_results['robustness'] = {
                'passed': passed,
                'value': model_metrics['robustness_consistency'],
                'threshold': self.config['min_robustness_consistency']
            }
            if not passed:
                all_passed = False
        
        # Calibration gate
        if 'max_calibration_ece' in self.config:
            passed = model_metrics['calibration_ece'] <= self.config['max_calibration_ece']
            gate_results['calibration'] = {
                'passed': passed,
                'value': model_metrics['calibration_ece'],
                'threshold': self.config['max_calibration_ece']
            }
            if not passed:
                all_passed = False
        
        # Regression gate
        if 'no_regressions' in self.config and self.config['no_regressions']:
            passed = model_metrics.get('regressions', 0) == 0
            gate_results['regressions'] = {
                'passed': passed,
                'value': model_metrics.get('regressions', 0),
                'threshold': 0
            }
            if not passed:
                all_passed = False
        
        # Print report
        print(f"\n{'='*60}")
        print("QUALITY GATE CHECK")
        print(f"{'='*60}")
        
        for gate, result in gate_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{gate.upper():<20} {status}")
            print(f"  Value: {result['value']:.4f}, Threshold: {result['threshold']:.4f}")
        
        print(f"\n{'='*60}")
        overall_status = "✅ READY FOR RELEASE" if all_passed else "❌ NOT READY FOR RELEASE"
        print(f"OVERALL: {overall_status}")
        print(f"{'='*60}")
        
        return {
            'all_passed': all_passed,
            'gate_results': gate_results
        }
    
    def generate_release_report(self, model_info, metrics, gate_results):
        """Generate comprehensive release report."""
        report = f"""
# Model Release Report

## Model Information
- **Name**: {model_info['name']}
- **Version**: {model_info['version']}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Training Data**: {model_info['training_data']}
- **Architecture**: {model_info['architecture']}

## Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1 Score**: {metrics['f1']:.4f}
- **AUC-ROC**: {metrics['auc_roc']:.4f}

## Quality Attributes
- **Fairness Disparity**: {metrics['fairness_disparity']:.4f}
- **Robustness Consistency**: {metrics['robustness_consistency']:.4f}
- **Calibration ECE**: {metrics['calibration_ece']:.4f}
- **Regressions**: {metrics.get('regressions', 0)}

## Quality Gate Results
"""
        
        for gate, result in gate_results['gate_results'].items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"- {gate.upper()}: {status}\n"
        
        report += f"""
## Release Decision
{'✅ APPROVED FOR RELEASE' if gate_results['all_passed'] else '❌ RELEASE BLOCKED'}

## Notes
{model_info.get('notes', 'No additional notes')}
"""
        
        return report
```

---

## Complete Validation Pipeline Example

```python
def complete_validation_pipeline(model, train_data, val_data, test_data, 
                                sensitive_attributes=None):
    """
    Run complete validation pipeline before model release.
    """
    print("="*70)
    print("COMPLETE MODEL VALIDATION PIPELINE")
    print("="*70)
    
    results = {}
    
    # 1. Basic Performance Evaluation
    print("\n[1/8] Basic Performance Evaluation")
    test_preds = model.predict(test_data['texts'])
    test_metrics = calculate_metrics(test_preds, test_data['labels'])
    results['basic_metrics'] = test_metrics
    
    # 2. Cross-Validation
    print("\n[2/8] Cross-Validation")
    cv_results, _ = k_fold_cross_validation(
        type(model), model.config, 
        train_data['texts'], train_data['labels'], 
        k=5
    )
    results['cross_validation'] = cv_results
    
    # 3. Robustness Testing
    print("\n[3/8] Robustness Testing")
    tester = RobustnessTester(model, model.tokenizer)
    robustness_results = tester.comprehensive_robustness_suite(
        test_data['texts'], test_data['labels']
    )
    results['robustness'] = robustness_results
    
    # 4. Fairness Audit (if sensitive attributes provided)
    if sensitive_attributes:
        print("\n[4/8] Fairness Audit")
        auditor = FairnessAuditor(test_preds, test_data['labels'], sensitive_attributes)
        fairness_report = auditor.generate_fairness_report()
        results['fairness'] = fairness_report
    else:
        print("\n[4/8] Fairness Audit: SKIPPED (no sensitive attributes)")
    
    # 5. Calibration Analysis
    print("\n[5/8] Calibration Analysis")
    calibrator = CalibrationAnalyzer(model)
    calibration_results = calibrator.plot_calibration_curve(
        test_data['texts'], test_data['labels']
    )
    results['calibration'] = calibration_results
    
    # 6. Domain Shift Detection (if target data available)
    # Skip for now
    
    # 7. Statistical Significance (if baseline available)
    # Skip for now
    
    # 8. Quality Gate Check
    print("\n[8/8] Quality Gate Check")
    
    # Prepare metrics for quality gates
    gate_metrics = {
        'accuracy': test_metrics['accuracy'],
        'fairness_disparity': max(
            [v['max_disparity'] for k, v in results.get('fairness', {}).items()]
        ) if 'fairness' in results else 0.0,
        'robustness_consistency': np.mean([
            v['consistency'] for v in robustness_results.values()
        ]),
        'calibration_ece': calibration_results['ece'],
        'regressions': 0  # Would check against baseline
    }
    
    release_config = {
        'min_accuracy': 0.80,
        'max_fairness_disparity': 0.15,
        'min_robustness_consistency': 0.85,
        'max_calibration_ece': 0.10,
        'no_regressions': True
    }
    
    gate_checker = QualityGateChecker(release_config)
    gate_results = gate_checker.check_all_gates(gate_metrics)
    results['quality_gates'] = gate_results
    
    # Final Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Basic Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"CV Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
    print(f"Avg Robustness: {gate_metrics['robustness_consistency']:.4f}")
    if 'fairness' in results:
        print(f"Max Fairness Disparity: {gate_metrics['fairness_disparity']:.4f}")
    print(f"Calibration ECE: {gate_metrics['calibration_ece']:.4f}")
    print(f"\nRelease Status: {'✅ APPROVED' if gate_results['all_passed'] else '❌ BLOCKED'}")
    
    return results

# Usage
# validation_results = complete_validation_pipeline(
#     model, train_data, val_data, test_data,
#     sensitive_attributes={'gender': gender_array, 'age': age_array}
# )
```

---

## Best Practices Checklist

### Pre-Release Validation Checklist

- [ ] **Data Splits**: Proper train/val/test separation with no leakage
- [ ] **Cross-Validation**: K-fold CV completed with consistent results
- [ ] **Statistical Power**: Test set size sufficient for desired confidence
- [ ] **Performance Metrics**: All primary metrics meet thresholds
- [ ] **Fairness Audit**: No significant bias across protected groups
- [ ] **Robustness Testing**: Model stable under perturbations
- [ ] **Calibration**: Predictions well-calibrated (ECE < threshold)
- [ ] **Edge Cases**: Critical edge cases handled correctly
- [ ] **Regression Tests**: No regressions from baseline
- [ ] **Documentation**: All validation results documented

### Continuous Validation

- [ ] **Automated Testing**: Validation suite runs on every commit
- [ ] **Monitoring**: Production performance tracked continuously
- [ ] **Drift Detection**: Data drift monitored and alerted
- [ ] **Periodic Re-evaluation**: Full validation quarterly
- [ ] **Incident Response**: Process for handling validation failures

---

## Next Steps

In the next tutorial, we'll cover:
- **Continual Learning**: Strategies for updating models with new data
- **Catastrophic Forgetting Prevention**: Techniques to retain old knowledge
- **Incremental Training**: Efficient updates without full retraining
- **Version Management**: Model versioning and rollback strategies
- **Production Deployment**: Serving, scaling, and monitoring
