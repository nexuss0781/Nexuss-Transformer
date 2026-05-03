# Tutorial 9: Release Management & Version Control

## Overview
Training a model is not the end—it's the beginning of its lifecycle. This tutorial covers **versioning strategies**, **model freezing**, **release protocols**, **rollback procedures**, and **deprecation policies**. We'll use NTF's `ModelRegistry` for semantic versioning and metadata tracking. Proper release management ensures reproducibility, accountability, and smooth operations in production.

## Prerequisites
- Understanding of training pipelines (Tutorial 02)
- Familiarity with validation metrics (Tutorial 07)
- Basic knowledge of Git version control

---

## 1. Semantic Versioning with ModelRegistry

NTF provides built-in semantic versioning through `ModelRegistry`. Adopt a clear versioning scheme to communicate changes effectively.

### Format: `MAJOR.MINOR.PATCH`

| Component | When to Increment | Example |
|-----------|-------------------|---------|
| **MAJOR** | Breaking changes, architecture shifts, domain changes | `1.0.0` → `2.0.0` (Llama-2 → Llama-3) |
| **MINOR** | New features, performance improvements, backward-compatible | `1.2.0` → `1.3.0` (added SQL capability) |
| **PATCH** | Bug fixes, retraining on more data, no architecture change | `1.2.3` → `1.2.4` (fixed tokenization bug) |

### Using NTF's ModelRegistry

```python
from ntf.models import ModelRegistry
from ntf.config import ModelConfig
from ntf.utils.versioning import ModelStage, create_model_metadata

# Initialize registry with versioning enabled
registry = ModelRegistry(
    model_config=ModelConfig(name="meta-llama/Llama-2-7b-hf"),
    registry_path="./model_registry",
    enable_versioning=True
)

# After training, save with automatic versioning
registry.save_model(
    model=trained_model,
    tokenizer=tokenizer,
    version="1.0.0",  # Semantic versioning
    metadata={
        "training_config": config.to_dict(),
        "metrics": {"eval_loss": 0.234, "perplexity": 12.5},
        "dataset": "custom_instructions_v1",
        "peft_method": "lora",
        "notes": "Initial fine-tuning run"
    }
)

# List all versions
versions = registry.list_versions()
print(f"Available versions: {versions}")

# Load specific version
model_v1, tokenizer = registry.load_model_and_tokenizer(version="1.0.0")

# Compare versions
comparison = registry.compare_versions(["1.0.0", "1.1.0"], metrics=["eval_loss"])

# Rollback to previous version if needed
registry.rollback("1.0.0")
```

### Additional Metadata

Use build metadata for internal tracking:
- `1.2.3+data2024Q1` - Trained on Q1 2024 data
- `1.2.3+rlhf` - Includes RLHF fine-tuning
- `1.2.3+hotfix1` - Emergency patch

### Version Naming Convention

```
{model-family}-{version}-{quantization}-{context-length}

Examples:
- nexuss-assistant-2.1.0-fp16-4k
- nexuss-assistant-2.1.0-int4-8k
- nexuss-coder-1.0.0-bf16-16k
```

### Semantic Versioning Guidelines

- **MAJOR.MINOR.PATCH** format (e.g., 1.0.0, 2.1.3)
- **MAJOR**: Breaking changes, architecture modifications
- **MINOR**: New features, performance improvements
- **PATCH**: Bug fixes, minor adjustments

### Metadata Best Practices

- Always include training configuration
- Document dataset version and preprocessing
- Record evaluation metrics
- Add notes about known limitations
- Tag production-ready models

---

## 2. Model Freezing Strategies

Once a model version is released for production, it must be **frozen** to ensure consistency.

### What Does "Freezing" Mean?
1. **Weights Frozen**: No further training or fine-tuning on this version.
2. **Config Frozen**: Inference parameters (temperature, top_p) documented and locked.
3. **Dependencies Frozen**: Exact library versions recorded (torch, transformers, etc.).
4. **Data Frozen**: Training dataset snapshot archived with checksums.

### Implementation: Checkpoint Locking

```python
import json
import hashlib
from pathlib import Path
from datetime import datetime

def freeze_model_checkpoint(checkpoint_path, version, output_dir):
    """
    Create an immutable, versioned release package.
    """
    checkpoint = Path(checkpoint_path)
    release_dir = Path(output_dir) / f"release-{version}"
    release_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy model weights
    import shutil
    for file in checkpoint.glob("*"):
        if file.is_file():
            shutil.copy2(file, release_dir / file.name)
    
    # 2. Generate checksums
    checksums = {}
    for file in release_dir.glob("*"):
        with open(file, 'rb') as f:
            checksums[file.name] = hashlib.sha256(f.read()).hexdigest()
    
    # 3. Create release manifest
    manifest = {
        "version": version,
        "freeze_date": datetime.now().isoformat(),
        "status": "frozen",
        "checksums": checksums,
        "environment": {
            "python_version": "3.10.12",
            "torch_version": "2.1.0",
            "transformers_version": "4.35.0",
            "cuda_version": "12.1"
        },
        "training_config": {
            "base_model": "meta-llama/Llama-2-7b-hf",
            "epochs": 3,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "dataset_sha256": "abc123..."
        },
        "evaluation_metrics": {
            "accuracy": 0.89,
            "f1_score": 0.87,
            "toxicity_score": 0.02
        },
        "approved_by": "ai-team-lead",
        "notes": "Production release for Q4 2024"
    }
    
    # 4. Save manifest (read-only)
    manifest_path = release_dir / "release_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # 5. Make directory read-only (Unix)
    import os
    for file in release_dir.glob("*"):
        os.chmod(file, 0o444)  # Read-only
    
    print(f"✅ Model frozen: {release_dir}")
    return release_dir

# Usage
freeze_model_checkpoint(
    checkpoint_path="outputs/checkpoint-final",
    version="2.1.0",
    output_dir="releases"
)
```

---

## 3. Release Protocol

A structured process to move models from development to production.

### Stage Gate Process

```
[Development] → [Staging] → [Canary] → [Production] → [Deprecated]
     ↓              ↓           ↓           ↓             ↓
  Unit Tests   Integration  5% Traffic  100% Traffic  Archive
```

### Step-by-Step Release Checklist

#### Phase 1: Pre-Release Validation
- [ ] All unit tests pass
- [ ] Performance metrics meet thresholds (Tutorial 07)
- [ ] Bias/fairness audit completed
- [ ] Security scan passed (no prompt injection vulnerabilities)
- [ ] Documentation updated (API docs, model card)
- [ ] Rollback plan documented

#### Phase 2: Staging Deployment
- [ ] Deploy to staging environment
- [ ] Run integration tests with real-world traffic patterns
- [ ] Verify latency SLAs (<200ms P99)
- [ ] Monitor for 24-48 hours

#### Phase 3: Canary Release
- [ ] Route 5% of production traffic to new version
- [ ] Compare metrics against baseline (A/B test)
- [ ] Monitor error rates, latency, user feedback
- [ ] If metrics degrade → Rollback immediately
- [ ] If stable for 48 hours → Increase to 25%, then 50%, then 100%

#### Phase 4: Full Production
- [ ] 100% traffic routing
- [ ] Update DNS/load balancer configs
- [ ] Notify stakeholders
- [ ] Archive previous version (but keep available for rollback)

### Automated Release Script

```python
# release_pipeline.py
import subprocess
import sys

def run_stage(stage_name, commands):
    print(f"\n🚀 Starting Stage: {stage_name}")
    for cmd in commands:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed at: {cmd}")
            print(result.stderr)
            return False
    print(f"✅ Stage {stage_name} completed")
    return True

# Define stages
stages = {
    "validation": [
        "pytest tests/ --cov=src",
        "python scripts/evaluate_model.py --min-accuracy 0.85",
        "python scripts/check_bias.py --threshold 0.1"
    ],
    "staging": [
        "kubectl apply -f k8s/staging-deployment.yaml",
        "sleep 300",  # Wait 5 minutes
        "python scripts/smoke_test.py --env staging"
    ],
    "canary": [
        "kubectl set image deployment/assistant assistant=my-registry/model:v2.1.0-canary",
        "kubectl rollout status deployment/assistant",
        "python scripts/ab_test_monitor.py --canary-percent 5"
    ],
    "production": [
        "kubectl set image deployment/assistant assistant=my-registry/model:v2.1.0",
        "kubectl rollout status deployment/assistant",
        "python scripts/notify_slack.py --message 'v2.1.0 deployed to production'"
    ]
}

# Execute pipeline
for stage_name, commands in stages.items():
    if not run_stage(stage_name, commands):
        print(f"\n⛔ Pipeline failed at {stage_name}. Initiating rollback...")
        subprocess.run("kubectl rollout undo deployment/assistant", shell=True)
        sys.exit(1)

print("\n🎉 Release completed successfully!")
```

---

## 4. Rollback Procedures

When things go wrong (and they will), you need a fast, reliable rollback mechanism.

### Rollback Triggers
- Error rate spikes > 5%
- Latency P99 exceeds SLA by 50%
- User complaints surge
- Data drift detected (Tutorial 08)
- Security vulnerability discovered

### Rollback Strategies

#### A. Immediate Rollback (Blue-Green)
Switch traffic back to previous version instantly.
```bash
# Kubernetes example
kubectl rollout undo deployment/assistant

# Or manually switch service selector
kubectl patch service assistant -p '{"spec":{"selector":{"version":"v2.0.0"}}}'
```

#### B. Gradual Rollback
If the issue is not critical, gradually reduce traffic.
- 100% → 50% → 25% → 0% on bad version
- Allows investigation while minimizing impact

#### C. Hotfix Patch
For minor issues, deploy a patch version quickly.
- `v2.1.0` (bad) → `v2.1.1` (fixed)
- Only if fix can be developed and tested in <1 hour

### Rollback Runbook Template

```markdown
# Rollback Runbook for Model v{version}

## Emergency Contacts
- On-call Engineer: +1-XXX-XXX-XXXX
- Team Lead: +1-XXX-XXX-XXXX

## Rollback Commands
1. Stop canary: `kubectl delete hpa assistant-canary`
2. Revert deployment: `kubectl rollout undo deployment/assistant`
3. Verify: `kubectl get pods -l app=assistant`

## Post-Rollback Actions
- [ ] Notify stakeholders
- [ ] Create incident report
- [ ] Schedule post-mortem
- [ ] Archive faulty release
```

---

## 5. Model Registry & Artifact Management

Centralize all model versions in a registry for tracking and access control.

### Using MLflow Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = "runs:/abc123/model"
result = client.create_registered_model("nexuss-assistant")

# Create version
version = client.create_model_version(
    name="nexuss-assistant",
    source=model_uri,
    run_id="abc123"
)

# Add aliases for easy reference
client.set_registered_model_alias("nexuss-assistant", "production", version.version)
client.set_registered_model_alias("nexuss-assistant", "staging", version.version)

# Load by alias (in production code)
model = mlflow.pytorch.load_model("models:/nexuss-assistant/production")
```

### Registry Metadata Requirements
Every registered model should include:
- **Version number** (semantic)
- **Git commit hash** (code version)
- **Dataset version** (data lineage)
- **Training metrics** (accuracy, loss curves)
- **Evaluation results** (benchmark scores)
- **Approver** (who signed off)
- **Known limitations** (what the model struggles with)

---

## 6. Deprecation & Retirement

Models don't last forever. Plan for end-of-life.

### Deprecation Timeline

| Phase | Duration | Actions |
|-------|----------|---------|
| **Announcement** | 30 days before | Notify users, document migration path |
| **Warning Period** | 30-60 days | Log warnings when old version used |
| **Read-Only** | 60-90 days | No new features, security patches only |
| **Retirement** | 90+ days | Shut down endpoints, archive artifacts |

### Deprecation Notice Template

```markdown
## ⚠️ Deprecation Notice: nexuss-assistant v1.5.0

**Deprecation Date**: January 15, 2025  
**End-of-Life Date**: April 15, 2025

### Reason
This version uses outdated tokenizer with known security vulnerabilities.

### Migration Path
Upgrade to v2.1.0:
```bash
kubectl set image deployment/assistant assistant=my-registry/nexuss-assistant:v2.1.0
```

### Support
Contact ai-support@company.com for migration assistance.
```

### Archival Process

```python
import tarfile
from pathlib import Path

def archive_release(release_path, archive_dir):
    """
    Compress and archive a retired release.
    """
    release = Path(release_path)
    archive = Path(archive_dir) / f"{release.name}-archived.tar.gz"
    
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(release, arcname=release.name)
    
    # Store in cold storage (S3 Glacier, etc.)
    # upload_to_glacier(archive)
    
    # Delete local copy
    import shutil
    shutil.rmtree(release)
    
    print(f"📦 Archived: {archive}")
    return archive
```

---

## 7. Practical Exercise: End-to-End Release

**Scenario**: Release a fine-tuned assistant model to production.

### Step 1: Freeze the Model
```bash
python scripts/freeze_model.py \
  --checkpoint outputs/final-checkpoint \
  --version 2.1.0 \
  --output releases/
```

### Step 2: Register in MLflow
```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")

with mlflow.start_run():
    mlflow.pytorch.log_model(
        model="releases/release-2.1.0",
        artifact_path="model",
        registered_model_name="nexuss-assistant"
    )
```

### Step 3: Run Release Pipeline
```bash
python release_pipeline.py
# Executes: validation → staging → canary → production
```

### Step 4: Monitor & Verify
```bash
# Check deployment status
kubectl get deployment assistant

# View logs
kubectl logs -l app=assistant --tail=100

# Test endpoint
curl -X POST https://api.company.com/v1/chat \
  -d '{"message": "Hello"}' \
  -H "Authorization: Bearer $TOKEN"
```

### Step 5: Document Release
Update `RELEASES.md`:
```markdown
## v2.1.0 (2024-10-15)

### Improvements
- 15% better accuracy on customer support queries
- Reduced toxicity score from 0.05 to 0.02
- Added support for multi-turn conversations

### Known Issues
- Struggles with highly technical jargon (planned for v2.2.0)

### Rollback Instructions
kubectl rollout undo deployment/assistant
```

---

## 8. Summary Checklist

| Task | Tool/Method | Frequency |
|------|-------------|-----------|
| **Version Numbering** | Semantic versioning | Every release |
| **Checkpoint Freezing** | SHA256 checksums, read-only perms | Pre-release |
| **Validation Gates** | Automated tests, bias checks | Pre-release |
| **Staging Deployment** | K8s, Docker | Every release |
| **Canary Testing** | 5% traffic, A/B tests | Every release |
| **Rollback Plan** | Runbook, automated scripts | Prepared in advance |
| **Registry Updates** | MLflow, Hugging Face Hub | Every release |
| **Deprecation Notices** | Documentation, emails | When retiring |
| **Archival** | Cold storage, compression | Post-retirement |

---

## Next Steps
In Tutorial 10, we will explore **Distributed Training at Scale**, covering Tensor Parallelism, Pipeline Parallelism, and ZeRO optimization for training massive models across multiple GPUs and nodes.
