# --------------------------------------------------
# Chatterbox TTS - RunPod Serverless
# https://github.com/resemble-ai/chatterbox
# --------------------------------------------------
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# --------------------------------------------------
# Environment variables
# --------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

# --------------------------------------------------
# System dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    wget \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Working directory
# --------------------------------------------------
WORKDIR /app

# --------------------------------------------------
# Install Python dependencies
# --------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip

# Completely reinstall PyTorch stack from official CUDA 12.4 wheels
# This fixes the torchvision::nms operator issue
RUN pip uninstall -y torch torchvision torchaudio || true && \
    pip install --no-cache-dir \
    torch==2.4.0+cu124 \
    torchvision==0.19.0+cu124 \
    torchaudio==2.4.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify torch installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); import torchvision; print(f'TorchVision {torchvision.__version__}')"

# Install Chatterbox TTS (without reinstalling torch)
RUN pip install --no-cache-dir --no-deps chatterbox-tts && \
    pip install --no-cache-dir \
    transformers \
    safetensors \
    einops \
    tokenizers \
    huggingface_hub \
    s3tokenizer \
    vocos \
    perth-watermarker

# Install additional dependencies
RUN pip install --no-cache-dir \
    runpod \
    boto3 \
    requests \
    soundfile \
    librosa \
    huggingface_hub[hf_transfer]

# --------------------------------------------------
# Pre-download model weights (speeds up cold starts)
# Using CPU during build since GPU may not be available
# Model weights are the same, just loaded to CUDA at runtime
# --------------------------------------------------
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')" || echo "Model pre-download skipped, will download on first request"

# --------------------------------------------------
# Copy handler
# --------------------------------------------------
COPY handler.py /app/handler.py

# --------------------------------------------------
# Create directories
# --------------------------------------------------
RUN mkdir -p /app/input /app/output /tmp/audio

# --------------------------------------------------
# Start RunPod Serverless
# --------------------------------------------------
CMD ["python", "/app/handler.py"]
