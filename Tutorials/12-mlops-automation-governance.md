# Tutorial 12: MLOps, Automation & Governance

## Overview
Moving from experimental notebooks to production requires robust **MLOps** practices. This tutorial covers the end-to-end lifecycle automation: **CI/CD for ML**, **Model Registries**, **Experiment Tracking**, **Compliance**, and **Automated Evaluation Gates**. In regulated industries (finance, healthcare), governance is as critical as model accuracy.

## Prerequisites
- Understanding of training pipelines (Tutorial 02)
- Basic knowledge of Docker and Git
- Familiarity with validation concepts (Tutorial 07)

---

## 1. The MLOps Lifecycle

Traditional DevOps focuses on code. MLOps adds two dynamic assets: **Data** and **Models**.

### Key Differences
| Aspect | DevOps | MLOps |
| :--- | :--- | :--- |
| **Versioning** | Code (Git) | Code + Data (DVC) + Models (Registry) |
| **Testing** | Unit/Integration tests | Data quality + Model performance + Bias checks |
| **Deployment** | Binary/Container | Model artifacts + Config + Dependencies |
| **Monitoring** | Latency/Error rates | Drift (Data/Concept) + Performance decay |

### The Three Levels of MLOps Maturity
1.  **Level 0 (Manual)**: Manual training, manual deployment. Scripts in notebooks.
2.  **Level 1 (Pipeline Automation)**: Automated retraining via CI/CD. Basic monitoring.
3.  **Level 2 (CI/CD/CT)**: Continuous Integration, Deployment, and **Training**. Triggered by data drift or code changes.

---

## 2. Experiment Tracking

Stop naming files `model_final_v2_new.py`. Use a tracker to log parameters, metrics, and artifacts.

### Tools
- **MLflow**: Open source, widely adopted.
- **Weights & Biases (W&B)**: Cloud-native, excellent visualization.
- **TensorBoard**: Basic, built-in to PyTorch/TF.

### Implementing MLflow
```python
import mlflow
import mlflow.pytorch
from transformers import AutoModelForCausalLM

# Set tracking server (local or remote)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Llama-FineTuning-Prod")

with mlflow.start_run(run_name="epoch-3-lr-1e-4"):
    # Log Parameters
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("model_name", "meta-llama/Llama-2-7b-hf")
    
    # Training Loop
    # ... train ...
    final_loss = 0.45
    eval_accuracy = 0.89
    
    # Log Metrics
    mlflow.log_metric("train_loss", final_loss)
    mlflow.log_metric("eval_accuracy", eval_accuracy)
    
    # Log Artifacts (The Model)
    model = AutoModelForCausalLM.from_pretrained("output_dir")
    mlflow.pytorch.log_model(model, "model")
    
    # Log Data Version
    mlflow.log_param("data_version", "v2.3-cleaned")
    
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
```

**Benefits**:
- Reproducibility: Know exactly which code/data produced a model.
- Comparison: Compare hyperparameters side-by-side.
- Promotion: Promote specific Run IDs to Production.

---

## 3. Model Registry & Versioning

A Model Registry is a centralized store for managing model versions, stages, and permissions.

### Semantic Versioning for Models
Use `MAJOR.MINOR.PATCH`:
- **MAJOR**: Architecture change or domain shift (e.g., Llama-2 → Llama-3).
- **MINOR**: Significant performance improvement or new features (e.g., added SQL capability).
- **PATCH**: Bug fix, retrain on slightly more data, same architecture.

### Stages
1.  **None**: Just logged.
2.  **Staging**: Passed automated tests, ready for QA.
3.  **Production**: Live serving traffic.
4.  **Archived**: Deprecated but kept for audit/revert.

### Programmatic Promotion
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "customer-support-bot"
run_id = "abc123..."

# Register model from run
model_uri = f"runs:/{run_id}/model"
result = client.create_registered_model(model_name)

# Create version
version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

# Transition to Staging
client.transition_model_version_stage(
    name=model_name,
    version=version.version,
    stage="Staging"
)

# After manual approval -> Production
# client.transition_model_version_stage(..., stage="Production")
```

---

## 4. CI/CD for Machine Learning

### Continuous Integration (CI)
Automated testing on every Pull Request.

**Pipeline Steps**:
1.  **Linting**: Check code style (`flake8`, `black`).
2.  **Unit Tests**: Test data loaders, loss functions, utility scripts.
3.  **Data Validation**: Check schema, missing values, distribution shifts (using Great Expectations).
4.  **Model Tests**: 
    - Overfit test: Can model memorize 10 samples? (Sanity check)
    - Performance threshold: Does new model beat baseline on validation set?

**.github/workflows/ml-ci.yml** (GitHub Actions Example):
```yaml
name: ML Pipeline CI

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt pytest great-expectations
      
      - name: Run Unit Tests
        run: pytest tests/unit/
      
      - name: Validate Data Schema
        run: python scripts/validate_data.py --path data/validation.json
      
      - name: Quick Train & Eval (Smoke Test)
        run: python train.py --epochs 1 --subset 1000 --min-accuracy 0.75
```

### Continuous Deployment (CD)
Automated deployment when a model reaches "Production" stage.

**Pipeline Steps**:
1.  **Build Container**: Dockerize the inference server (vLLM/TGI).
2.  **Push to Registry**: ECR, Docker Hub.
3.  **Deploy to Kubernetes**: Update deployment manifest with new image/model version.
4.  **Canary Rollout**: Route 5% traffic to new model. Monitor errors/latency.
5.  **Full Rollout**: If metrics stable, increase to 100%.

---

## 5. Automated Evaluation Gates

Never deploy without passing strict gates. These should be part of the CD pipeline.

### Gate Types
1.  **Performance Gate**: Accuracy/F1 must be $\ge$ Baseline.
2.  **Regression Gate**: Must not degrade performance on critical slices (e.g., "short texts", "non-English").
3.  **Fairness Gate**: Disparate impact ratio must be within threshold (e.g., < 1.25).
4.  **Safety Gate**: Toxicity score < 0.05 on adversarial dataset.
5.  **Latency Gate**: P99 inference time < 200ms on target hardware.

### Implementation Example
```python
# evaluation_gate.py
import sys
import json

def check_gates(metrics, thresholds):
    failed = []
    for metric, value in metrics.items():
        if metric in thresholds:
            op, limit = thresholds[metric]
            if op == ">=" and value < limit:
                failed.append(f"{metric} ({value}) < {limit}")
            if op == "<=" and value > limit:
                failed.append(f"{metric} ({value}) > {limit}")
    
    if failed:
        print("❌ GATES FAILED:")
        for f in failed: print(f"  - {f}")
        return False
    else:
        print("✅ ALL GATES PASSED")
        return True

if __name__ == "__main__":
    # Load metrics from eval run
    current_metrics = json.load(open("eval_results.json"))
    
    thresholds = {
        "accuracy": (">=", 0.85),
        "toxicity_score": ("<=", 0.05),
        "p99_latency_ms": ("<=", 200),
        "fairness_ratio": ("<=", 1.2)
    }
    
    success = check_gates(current_metrics, thresholds)
    sys.exit(0 if success else 1) # Exit code 1 breaks CI/CD
```

---

## 6. Governance & Compliance

In regulated sectors, you must answer: *Why did the model make this decision?* and *Who approved this version?*

### Model Cards
Document every model. Include:
- **Intended Use**: What is it for? What is it NOT for?
- **Training Data**: Sources, preprocessing, known biases.
- **Evaluation Results**: Benchmarks, limitations.
- **Ethical Considerations**: Potential misuse.

*Example Snippet*:
```markdown
## Intended Use
This model is designed for summarizing customer support tickets in English.
**Out-of-scope**: Medical advice, legal interpretation, non-English languages.

## Training Data
- Source: Internal support logs (2022-2023).
- Preprocessing: PII redaction applied.
- Bias: Underrepresented dialects may have lower accuracy.
```

### Audit Trails
Log every prediction (or a statistical sample) with:
- Input hash
- Model Version ID
- Timestamp
- User ID (if applicable)
- Output

This allows retroactive analysis if a bug or bias is discovered later.

### Data Privacy (GDPR/CCPA)
- **Right to be Forgotten**: Can you remove a user's data from the model?
  - *Challenge*: Hard to "unlearn" from neural nets.
  - *Solution*: Maintain data lineage. If deletion requested, flag data, retrain model excluding that data (Machine Unlearning techniques are emerging but retraining is standard).
- **PII Detection**: Scan training data and inputs/outputs for PII using tools like Microsoft Presidio.

---

## 7. Monitoring in Production

Deployment is not the end. Models decay.

### Types of Drift
1.  **Data Drift**: Input distribution changes ($P(X)$ changes).
    - *Example*: User queries shift from "desktop support" to "mobile app" after a new release.
    - *Detection*: Statistical tests (KS-test, PSI) on input features.
2.  **Concept Drift**: Relationship between input and output changes ($P(Y|X)$ changes).
    - *Example*: "Sick" meant "ill" in 2019, but "cool" in 2024 slang. Model predictions become wrong despite same input stats.
    - *Detection*: Monitor accuracy (requires ground truth labels, often delayed).
3.  **Upstream Drift**: Data pipeline breaks.
    - *Example*: A sensor sends -1 instead of null. Column missing.
    - *Detection*: Schema validation, null checks.

### Monitoring Stack Architecture
1.  **Inference Service**: Logs predictions to a message queue (Kafka/Kinesis).
2.  **Drift Detector**: Consumes logs, computes statistics (PSI, KL divergence) against reference data.
3.  **Dashboard**: Grafana/Prometheus for system metrics; Evidently AI/Arize for data quality.
4.  **Alerting**: Slack/PagerDuty alert if drift > threshold.

```python
# Evidently AI Example for Drift Detection
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.column_mapping import ColumnMapping

# Reference data (training) vs Current data (production window)
report = Report(metrics=[DataDriftTable()])
report.run(reference_data=ref_df, current_data=prod_df)
report.save_html("drift_report.html")
# If drift detected -> Trigger retraining pipeline
```

---

## 8. Continuous Training (CT)

Automate the retraining loop.

**Triggers**:
- **Schedule**: Weekly/Monthly retrain.
- **Event**: New major dataset available.
- **Metric**: Performance drops below threshold (Drift detected).

**CT Pipeline Flow**:
1.  **Trigger**: Drift alert or Schedule.
2.  **Data Prep**: Fetch latest labeled data. Validate schema.
3.  **Train**: Launch training job (SageMaker, Vertex AI, K8s Job).
4.  **Evaluate**: Run evaluation suite against holdout set.
5.  **Gate**: Check automated gates.
6.  **Register**: If passed, register new version in Model Registry.
7.  **Deploy**: Canary deployment to staging, then production.

---

## 9. Practical Exercise: Building a CI/CD Pipeline

**Scenario**: Automate the deployment of a sentiment analysis model.

**Step 1: Define Tests (`tests/test_model.py`)**
```python
def test_model_performance():
    model = load_model("staging")
    acc = evaluate(model, val_dataset)
    assert acc > 0.80, "Model accuracy regressed!"
```

**Step 2: GitHub Actions Workflow (`.github/workflows/deploy.yml`)**
```yaml
name: Deploy Model

on:
  workflow_run:
    workflows: ["ML Pipeline CI"]
    types:
      - completed

jobs:
  deploy:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v2
      
      - name: Build Docker Image
        run: docker build -t my-registry/sentiment-model:${{ github.sha }} .
      
      - name: Push to Registry
        run: docker push my-registry/sentiment-model:${{ github.sha }}
      
      - name: Deploy to Kubernetes (Canary)
        run: |
          kubectl set image deployment/sentiment-api \
          sentiment-container=my-registry/sentiment-model:${{ github.sha }} \
          --record
          kubectl rollout status deployment/sentiment-api
```

**Step 3: Post-Deployment Verification**
Run a smoke test against the live endpoint.
```bash
curl -X POST https://api.example.com/predict \
  -d '{"text": "I love this product"}' \
  -H "Content-Type: application/json"
# Verify response time < 200ms and valid JSON
```

---

## 10. Summary Checklist

| Component | Tool Examples | Purpose |
| :--- | :--- | :--- |
| **Tracking** | MLflow, W&B | Log params, metrics, artifacts |
| **Registry** | MLflow Registry, Hugging Face Hub | Versioning, staging, promotion |
| **CI** | GitHub Actions, Jenkins | Linting, unit tests, data validation |
| **CD** | ArgoCD, Spinnaker, K8s | Container deployment, canary rolls |
| **Evaluation** | Custom scripts, Evidently AI | Performance gates, bias checks |
| **Monitoring** | Prometheus, Grafana, Arize | Drift detection, latency tracking |
| **Governance** | Model Cards, Audit Logs | Compliance, explainability |

## Next Steps
In Tutorial 13, we will cover **Troubleshooting & Performance Debugging**, diving deep into diagnosing NaNs, OOM errors, convergence failures, and slow training issues.
