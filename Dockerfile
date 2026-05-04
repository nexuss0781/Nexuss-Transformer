# Dockerfile for Nexuss Transformer Framework - Blank Slate Training
# Optimized for Hugging Face Spaces with GPU support
# Modernized with stable CUDA 12.8 and compatible dependencies

FROM nvidia/cuda:12.8.2-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/tmp/huggingface
ENV PYTORCH_CUDA_VERSION=12.4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with pinned compatible versions
# Using specific versions to avoid pyarrow/datasets compatibility issues
RUN pip3 install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    transformers==4.44.0 \
    accelerate==0.33.0 \
    peft==0.12.0 \
    trl==0.9.6 \
    datasets==2.20.0 \
    pyarrow==16.1.0 \
    pandas==2.2.2 \
    sentencepiece==0.2.0 \
    safetensors==0.4.3 \
    huggingface_hub==0.24.5 \
    tokenizers==0.19.1 \
    numpy==1.26.4 \
    scipy==1.14.0 \
    tqdm==4.66.4 \
    pyyaml==6.0.1 \
    omegaconf==2.3.0 \
    nltk==3.8.1 \
    rouge_score==0.1.2 \
    sacrebleu==2.4.2 \
    tensorboard==2.17.0 \
    bitsandbytes==0.43.3 \
    ethiobbpe>=1.0.0

# Optional: Install Flash Attention 2 for Ampere+ GPUs (uncomment if using A100/H100/RTX30xx+)
# RUN pip install flash-attn==2.6.3 --no-build-isolation

# Copy the entire project
COPY . .

# Make training script executable
RUN chmod +x train_blank_slate.py

# Create directories for checkpoints and outputs
RUN mkdir -p checkpoints outputs logs

# Default command - can be overridden
CMD ["python3", "train_blank_slate.py", "--model_size", "small", "--num_epochs", "1", "--batch_size", "8"]

# Expose Gradio port if running inference UI
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1
