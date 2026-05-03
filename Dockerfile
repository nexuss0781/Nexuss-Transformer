# Dockerfile for Nexuss Transformer Framework - Blank Slate Training
# Optimized for Hugging Face Spaces with GPU support

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/tmp/huggingface

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

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    accelerate \
    datasets \
    pandas \
    pyarrow \
    sentencepiece \
    gradio \
    huggingface_hub

# Optional: Install Flash Attention 2 for Ampere+ GPUs (uncomment if using A100/H100/RTX30xx+)
# RUN pip install flash-attn --no-build-isolation

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
