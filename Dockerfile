# MODO S1 — Production Dockerfile for H100/A100 deployment
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for audio/video
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    git curl wget ffmpeg libsndfile1 libsox-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install vLLM (may need to build from source for latest)
RUN pip install vllm --no-build-isolation

# Install audio/video extras
RUN pip install \
    whisper@git+https://github.com/openai/whisper.git \
    kokoro-onnx soundfile \
    diffusers[flux,ltx] transformers accelerate \
    imageio imageio-ffmpeg

# App code
WORKDIR /app
COPY . /app

# Model cache dir
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
RUN mkdir -p /models

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server.py"]