#!/usr/bin/env bash
# MODO S1 — Deploy to GPU cloud (RunPod / Vast.ai)
# Usage: ./deploy.sh [runpod|vast] [GPU_TYPE]

set -euo pipefail

PROVIDER="${1:-runpod}"
GPU_TYPE="${2:-H100}"
REPO="skmandal3240/MODO-S1"  # Update this
IMAGE="ghcr.io/${REPO}:latest"
CONTAINER_NAME="modo-s1-api"
PORT=8000

echo "[MODO S1] Building Docker image..."
docker build -t "$IMAGE" .

echo "[MODO S1] Pushing to GHCR..."
docker push "$IMAGE"

if [[ "$PROVIDER" == "runpod" ]]; then
    echo "[MODO S1] Deploying to RunPod..."
    # Requires: pip install runpodctl && runpodctl config set apiKey YOUR_KEY
    runpodctl create pod \
        --name "$CONTAINER_NAME" \
        --image "$IMAGE" \
        --gpu-type "$GPU_TYPE" \
        --gpu-count 1 \
        --ports "8000/http" \
        --env MODO_MODEL=nvidia/Nemotron-3-Ultra \
        --env MODO_ADAPTER=/app/adapters/modo-s1-final-merged \
        --volume-name modo-models:/models \
        --volume-name modo-adapters:/app/adapters
    
    echo "[MODO S1] RunPod pod created. Get endpoint with: runpodctl get pod $CONTAINER_NAME"

elif [[ "$PROVIDER" == "vast" ]]; then
    echo "[MODO S1] Deploying to Vast.ai..."
    # Requires: pip install vastai && vastai set api-key YOUR_KEY
    vastai create instance \
        --image "$IMAGE" \
        --gpu "$GPU_TYPE" \
        --disk 100 \
        --env MODO_MODEL=nvidia/Nemotron-3-Ultra \
        --env MODO_ADAPTER=/app/adapters/modo-s1-final-merged \
        --onstart "cd /app && python server.py"
    
    echo "[MODO S1] Vast.ai instance created. Get SSH info with: vastai show instances"

else
    echo "Unknown provider: $PROVIDER. Use 'runpod' or 'vast'"
    exit 1
fi

echo ""
echo "[MODO S1] Deployment initiated!"
echo "[MODO S1] API will be at: https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions"
echo "[MODO S1] Or SSH into Vast instance: curl localhost:8000/v1/chat/completions"