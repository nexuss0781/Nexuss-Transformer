# NTF Deployment Templates

Production-ready deployment templates for serving fine-tuned transformer models.

## Available Templates

### Local Deployment
- **FastAPI** (`local/fastapi/`) - Modern, high-performance async API
- **Flask** (`local/flask/`) - Simple, lightweight API

### Cloud Platforms
- **AWS SageMaker** (`aws/sagemaker/`) - Managed ML service
- **AWS EC2** (`aws/ec2/`) - Self-managed on EC2
- **GCP Vertex AI** (`gcp/vertex-ai/`) - Google's managed ML platform
- **GCP GKE** (`gcp/gke/`) - Kubernetes on GCP
- **Azure AKS** (`azure/aks/`) - Kubernetes on Azure
- **Azure ML Endpoints** (`azure/ml-endpoints/`) - Azure ML managed endpoints

## Quick Start - FastAPI (Local)

```bash
cd deployment-templates/local/fastapi

# Install dependencies
pip install -r requirements.txt

# Set model path or name
export NTF_MODEL_PATH=./models  # or
export NTF_MODEL_NAME=meta-llama/Llama-2-7b-hf

# Run server
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- `GET /health` - Health check
- `POST /generate` - Text generation
- `POST /classify` - Text classification
- `GET /model/info` - Model information

### Example Request

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning in simple terms:",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NTF_MODEL_PATH` | Path to local model registry | None |
| `NTF_MODEL_NAME` | HuggingFace model name | None |
| `PORT` | Server port | 8000 |
| `HOST` | Server host | 0.0.0.0 |

## Security Best Practices

1. **Authentication**: Add API key or JWT authentication
2. **Rate Limiting**: Implement request rate limiting
3. **Input Validation**: Validate all input lengths and content
4. **HTTPS**: Always use HTTPS in production
5. **Model Isolation**: Run models in isolated containers

## Scaling Recommendations

- Use batch inference for high-throughput scenarios
- Enable model quantization for reduced memory usage
- Consider using optimized inference engines (vLLM, TGI)
- Implement caching for repeated prompts

## Monitoring

Integrate with monitoring tools:
- Prometheus + Grafana for metrics
- Jaeger for distributed tracing
- ELK stack for logging
