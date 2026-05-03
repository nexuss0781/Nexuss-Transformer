# Tutorial 013: Model Versioning and Release Management

## 📌 Overview

**Level**: Advanced  
**Duration**: 45 minutes  
**Prerequisites**: Tutorials 001-012 completed

Learn professional model versioning, release management, and production deployment strategies.

---

## 🎯 Learning Objectives

By the end of this tutorial, you will:
- Implement semantic versioning for models
- Manage model registry and metadata
- Create production releases
- Track model lineage and provenance
- Archive and deprecate old versions

---

## 1. Why Model Versioning Matters

### The Problem Without Versioning

```
❌ "Which model is in production?"
❌ "What changed in the latest update?"
❌ "Can we rollback to last week's model?"
❌ "Which dataset was this trained on?"
❌ "Who approved this release?"
```

### Benefits of Proper Versioning

✅ **Reproducibility**: Exact recreation of any release  
✅ **Accountability**: Clear ownership and approval chain  
✅ **Rollback**: Quick recovery from bad releases  
✅ **Compliance**: Audit trails for regulated industries  
✅ **Collaboration**: Clear communication across teams  

---

## 2. Semantic Versioning for ML Models

### Version Format: MAJOR.MINOR.PATCH

```
v2.5.3
│  │  └─ PATCH: Bug fixes, no behavior change
│  └──── MINOR: New features, backward compatible
└─────── MAJOR: Breaking changes, architecture changes
```

### When to Increment Each

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Fixed tokenization bug | PATCH | 1.2.3 → 1.2.4 |
| Improved perplexity | MINOR | 1.2.3 → 1.3.0 |
| Changed architecture | MAJOR | 1.2.3 → 2.0.0 |
| Trained on more data | MINOR | 1.2.3 → 1.3.0 |
| Changed vocabulary | MAJOR | 1.2.3 → 2.0.0 |

### Implementation

```python
from utils.versioning import ModelVersion

# Create version
version = ModelVersion(major=1, minor=2, patch=3)
print(f"Current version: {version}")  # v1.2.3

# Increment based on changes
version.increment_patch()   # v1.2.4
version.increment_minor()   # v1.3.0
version.increment_major()   # v2.0.0

# Parse from string
v = ModelVersion.from_string("2.5.3")
print(f"Parsed: {v.major}.{v.minor}.{v.patch}")
```

---

## 3. Model Registry Setup

### Initialize Registry

```python
from utils.versioning import ModelRegistry, ModelMetadata, ModelStage

# Create registry
registry = ModelRegistry(registry_path="./model_registry")

print("Registry initialized")
print(f"Models directory: {registry.models_dir}")
print(f"Metadata directory: {registry.metadata_dir}")
print(f"Releases directory: {registry.releases_dir}")
```

### Directory Structure

```
model_registry/
├── index.json              # Master index of all models
├── models/                 # Actual model files
│   ├── nexuss-base/
│   │   ├── 1.0.0/
│   │   ├── 1.1.0/
│   │   └── 2.0.0/
│   └── nexuss-chat/
│       └── 1.0.0/
├── metadata/               # Version metadata
│   ├── nexuss-base/
│   │   ├── 1.0.0.json
│   │   ├── 1.1.0.json
│   │   └── 2.0.0.json
│   └── nexuss-chat/
│       └── 1.0.0.json
└── releases/               # Production releases
    └── nexuss-base/
        └── 2.0.0/
            ├── model.safetensors
            ├── config.json
            ├── release_manifest.json
            └── README.md
```

---

## 4. Creating Model Metadata

### Complete Metadata Example

```python
from utils.versioning import create_model_metadata, ModelStage

metadata = create_model_metadata(
    name="nexuss-base",
    version="1.0.0",
    stage=ModelStage.DEVELOPMENT,
    description="Base language model with 350M parameters",
    tags=["base-model", "english", "general-purpose"],
    training_config={
        "learning_rate": 5e-4,
        "batch_size": 256,
        "epochs": 10,
        "optimizer": "adamw",
    },
    evaluation_metrics={
        "perplexity": 15.2,
        "accuracy": 0.72,
        "mmlu_score": 0.45,
    },
)

# Add architecture details
metadata.num_parameters = 350_000_000
metadata.hidden_size = 1024
metadata.num_layers = 24
metadata.num_heads = 16
metadata.vocab_size = 50257
metadata.max_position_embeddings = 2048

# Training information
metadata.training_dataset = "CommonCrawl + Wikipedia"
metadata.training_steps = 100_000
metadata.training_loss = 1.85
metadata.validation_loss = 1.92

# Lineage tracking
metadata.parent_version = None  # First version
metadata.fine_tuned_from = None  # Trained from scratch

# Changelog
metadata.changelog = [
    "Initial release",
    "Trained on 500B tokens",
    "Supports 2048 context length",
]
```

---

## 5. Registering a Model

### Complete Registration Workflow

```python
from models.config import NTFConfig
from models.transformer import NexussTransformer

# Train your model (from previous tutorials)
config = NTFConfig.medium()
model = NexussTransformer(config)
# ... training code ...

# Assume tokenizer exists
tokenizer = your_tokenizer

# Register the model
model_path = registry.register_model(
    model=model,
    tokenizer=tokenizer,
    config=config,
    metadata=metadata,
)

print(f"Model registered at: {model_path}")
```

### What Happens During Registration

1. ✅ Model weights saved
2. ✅ Configuration saved
3. ✅ Tokenizer saved
4. ✅ SHA256 hash computed
5. ✅ Parameter count verified
6. ✅ Metadata stored as JSON
7. ✅ Index updated

---

## 6. Model Promotion Workflow

### Stage Transitions

```
EXPERIMENTAL → DEVELOPMENT → STAGING → PRODUCTION → ARCHIVED
     ↓              ↓            ↓           ↓           ↓
  Research      Internal      Testing    Live      Deprecated
```

### Promote Through Stages

```python
from utils.versioning import ModelStage

# After initial training
registry.promote_model(
    name="nexuss-base",
    version="1.0.0",
    new_stage=ModelStage.DEVELOPMENT,
    changelog_entry="Initial training complete, ready for internal testing",
)

# After internal validation
registry.promote_model(
    name="nexuss-base",
    version="1.0.0",
    new_stage=ModelStage.STAGING,
    changelog_entry="Passed internal benchmarks, ready for staging",
)

# After stakeholder approval
registry.promote_model(
    name="nexuss-base",
    version="1.0.0",
    new_stage=ModelStage.PRODUCTION,
    changelog_entry="Approved for production deployment",
)
```

---

## 7. Creating Production Releases

### Create Release Package

```python
release_path = registry.create_release(
    name="nexuss-base",
    version="2.0.0",
    release_notes="""
    ## Major Updates
    
    - Upgraded to transformer architecture v2
    - Extended context length to 4096 tokens
    - Improved perplexity by 15%
    
    ## Breaking Changes
    
    - Vocabulary expanded from 50k to 52k tokens
    - Not compatible with v1.x adapters
    """,
    tags=["production", "major-release", "improved"],
)

print(f"Release created at: {release_path}")
```

### Release Contents

The release package includes:
- Model weights (`.safetensors` or `.bin`)
- Configuration (`config.json`)
- Tokenizer files
- Release manifest
- README with usage instructions
- Checksums for verification

---

## 8. Querying the Registry

### List All Models

```python
# Get all registered models
all_models = registry.list_models()

for model_name, versions in all_models.items():
    print(f"\n{model_name}:")
    for version in versions:
        print(f"  - {version}")
```

### Get Latest Version

```python
latest = registry.get_latest_version("nexuss-base")
print(f"Latest version: {latest}")  # e.g., "2.0.0"
```

### Get Version History

```python
history = registry.get_version_history("nexuss-base")

for version_info in history:
    print(f"\nVersion {version_info['version']}:")
    print(f"  Stage: {version_info['stage']}")
    print(f"  Created: {version_info['created_at']}")
    print(f"  Parameters: {version_info['num_parameters']:,}")
    print(f"  Perplexity: {version_info['evaluation_metrics'].get('perplexity', 'N/A')}")
```

### Load Specific Version

```python
# Get model path and metadata
model_path, metadata = registry.get_model(
    name="nexuss-base",
    version="1.5.0",  # Specify exact version
)

print(f"Loading model from: {model_path}")
print(f"Description: {metadata['description']}")
```

---

## 9. Model Lineage Tracking

### Track Fine-Tuning Lineage

```python
# Base model
base_metadata = create_model_metadata(
    name="nexuss-base",
    version="1.0.0",
    fine_tuned_from=None,
)

# Chat fine-tune
chat_metadata = create_model_metadata(
    name="nexuss-chat",
    version="1.0.0",
    fine_tuned_from="nexuss-base/1.0.0",  # Track parent
    parent_version="1.0.0",
    description="Chat-optimized version fine-tuned from base model",
)

# Code fine-tune
code_metadata = create_model_metadata(
    name="nexuss-code",
    version="1.0.0",
    fine_tuned_from="nexuss-base/1.0.0",  # Same parent
    description="Code generation specialist",
)
```

### Visualize Lineage

```
nexuss-base/1.0.0
├── nexuss-chat/1.0.0
│   └── nexuss-chat/1.1.0
├── nexuss-code/1.0.0
└── nexuss-medical/1.0.0
```

---

## 10. Archiving and Deprecation

### Archive Old Versions

```python
# Archive outdated version
registry.archive_model(
    name="nexuss-base",
    version="1.0.0",
    reason="Superseded by v2.0.0 with improved architecture",
)

# Check status
_, metadata = registry.get_model("nexuss-base", "1.0.0")
print(f"Status: {metadata['stage']}")  # "archived"
```

### Deprecation Policy

```python
def check_deprecation_status(model_name, version):
    """Check if model version is deprecated."""
    _, metadata = registry.get_model(model_name, version)
    
    if metadata['stage'] == 'archived':
        print(f"⚠️  WARNING: {model_name} v{version} is archived")
        print(f"   Reason: {metadata['changelog'][-1]}")
        return True
    
    return False
```

---

## 11. Production Integration

### CI/CD Pipeline Integration

```yaml
# .github/workflows/release.yml
name: Model Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Validate Model
        run: python scripts/validate_model.py
      
      - name: Register Model
        run: python scripts/register_model.py
      
      - name: Run Evaluation
        run: python scripts/evaluate.py
      
      - name: Create Release
        if: success()
        run: python scripts/create_release.py
      
      - name: Deploy to Staging
        if: success()
        run: python scripts/deploy_staging.py
```

### Loading in Production

```python
class ModelLoader:
    def __init__(self, registry_path: str):
        self.registry = ModelRegistry(registry_path)
        self.current_model = None
        self.current_version = None
    
    def load_production_model(self, model_name: str):
        """Load latest production model."""
        latest = self.registry.get_latest_version(model_name)
        
        model_path, metadata = self.registry.get_model(
            model_name, 
            latest
        )
        
        # Verify it's production-ready
        if metadata['stage'] != 'production':
            raise ValueError(f"Model not in production: {metadata['stage']}")
        
        # Load model
        self.current_model = AutoModel.from_pretrained(model_path)
        self.current_version = latest
        
        print(f"Loaded {model_name} v{latest}")
    
    def rollback(self, model_name: str, target_version: str):
        """Rollback to specific version."""
        # Verify version exists
        _, metadata = self.registry.get_model(model_name, target_version)
        
        # Load previous version
        self.load_specific_version(model_name, target_version)
        print(f"Rolled back to v{target_version}")
```

---

## 12. Best Practices

### Versioning Guidelines

✅ **DO**:
- Use semantic versioning consistently
- Document all breaking changes
- Keep metadata up-to-date
- Tag releases with git commits
- Maintain changelog

❌ **DON'T**:
- Skip version numbers
- Reuse version numbers
- Release without testing
- Forget to update metadata
- Delete old versions immediately

### Release Checklist

```markdown
## Pre-Release Checklist

- [ ] All tests passing
- [ ] Evaluation metrics meet thresholds
- [ ] Documentation updated
- [ ] Changelog complete
- [ ] Backward compatibility verified (or documented breaks)
- [ ] Security scan complete
- [ ] Stakeholder approval obtained

## Post-Release Checklist

- [ ] Deployment successful
- [ ] Monitoring active
- [ ] Rollback plan ready
- [ ] Users notified (if breaking)
- [ ] Old versions marked for archival
```

---

## 13. Real-World Example

### Complete Release Workflow

```python
# Scenario: Release new chat model version

# 1. Train and evaluate
model = train_chat_model()
metrics = evaluate_model(model)

# 2. Create metadata
metadata = create_model_metadata(
    name="nexuss-chat",
    version="2.1.0",  # Minor update
    stage=ModelStage.DEVELOPMENT,
    description="Improved helpfulness and safety",
    fine_tuned_from="nexuss-base/2.0.0",
    evaluation_metrics=metrics,
    changelog=[
        "Improved response quality",
        "Reduced harmful outputs by 40%",
        "Added support for 10 new languages",
    ],
)

# 3. Register
registry.register_model(model, tokenizer, config, metadata)

# 4. Promote through stages
registry.promote_model("nexuss-chat", "2.1.0", ModelStage.STAGING)

# Run staging tests...
if staging_tests_pass():
    # 5. Promote to production
    registry.promote_model("nexuss-chat", "2.1.0", ModelStage.PRODUCTION)
    
    # 6. Create release
    registry.create_release(
        "nexuss-chat",
        "2.1.0",
        release_notes="Improved chat capabilities",
    )
    
    # 7. Archive old version
    registry.archive_model(
        "nexuss-chat",
        "2.0.0",
        "Superseded by v2.1.0",
    )
```

---

## 📚 Summary

### Key Takeaways

✅ **Semantic versioning** provides clear communication  
✅ **Model registry** centralizes version management  
✅ **Metadata tracking** ensures reproducibility  
✅ **Stage promotion** enforces quality gates  
✅ **Lineage tracking** maintains audit trail  

### Quick Reference

```python
# Standard workflow
registry = ModelRegistry()
metadata = create_model_metadata(name, version, ...)
registry.register_model(model, tokenizer, config, metadata)
registry.promote_model(name, version, ModelStage.PRODUCTION)
registry.create_release(name, version, notes)
```

---

**Congratulations!** You've mastered model versioning and release management!

➡️ **Next**: [Tutorial 014: Continual Learning](./014_continual_learning.md)
