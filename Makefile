# MODO S1 — Makefile for training and deployment

.PHONY: help data train sft dpo merge deploy test clean

# Default target
help:
	@echo "MODO S1 — Commercial Multimodal AI API"
	@echo ""
	@echo "Targets:"
	@echo "  make data        - Prepare training data (scientific + general)"
	@echo "  make sft         - LoRA fine-tune Nemotron 3 Ultra (SFT)"
	@echo "  make dpo         - DPO personality alignment (badass style)"
	@echo "  make merge       - Merge final model for vLLM deployment"
	@echo "  make train       - Full pipeline: data -> sft -> dpo -> merge"
	@echo "  make deploy      - Build Docker image and deploy to GPU cloud"
	@echo "  make test        - Test API locally"
	@echo "  make clean       - Clean build artifacts"

# Configuration
BASE_MODEL = nvidia/Nemotron-3-Ultra
SFT_ADAPTER = adapters/modo-s1-sft
DPO_ADAPTER = adapters/modo-s1-dpo
FINAL_MODEL = adapters/modo-s1-final-merged
DATA_FILE = data/train.jsonl
DPO_DATA = data/dpo_preferences.jsonl

# 1. Prepare training data
data: $(DATA_FILE) $(DPO_DATA)

$(DATA_FILE):
	python prepare_data.py --out $(DATA_FILE) --max-samples 500000 --scientific-ratio 0.3

$(DPO_DATA):
	python generate_dpo.py --out $(DPO_DATA) --samples 5000

# 2. SFT LoRA training (requires GPU)
sft: data
	python train_lora.py --base $(BASE_MODEL) --data $(DATA_FILE) --out $(SFT_ADAPTER) --epochs 3 --batch 2

# 3. DPO personality alignment (requires GPU, run after SFT)
dpo: sft
	python train_dpo.py --base $(BASE_MODEL) --adapter $(SFT_ADAPTER) --pref-data $(DPO_DATA) --out $(DPO_ADAPTER) --epochs 1

# 4. Merge final model
merge: dpo
	python -c "\
from unsloth import FastLanguageModel; \
m,t=FastLanguageModel.from_pretrained(model_name='$(BASE_MODEL)', max_seq_length=8192, dtype='bfloat16'); \
m.load_adapter('$(DPO_ADAPTER)', adapter_name='final'); m.set_adapter('final'); \
m.save_pretrained_merged('$(FINAL_MODEL)', t, save_method='merged_16bit')"

# 5. Full training pipeline
train: merge
	@echo "Training complete. Final model at $(FINAL_MODEL)"

# 6. Docker build
docker-build:
	docker build -t modo-s1:latest .

# 7. Deploy to GPU cloud (runpod / vast.ai)
deploy-runpod: docker-build
	@echo "Deploy to RunPod: https://runpod.io/console/deploy"
	@echo "Image: modo-s1:latest"
	@echo "GPU: H100 (80GB) or A100 (80GB)"
	@echo "Container Disk: 50GB"
	@echo "Env vars: MODO_MODEL=$(BASE_MODEL), MODO_ADAPTER=/app/$(FINAL_MODEL)"

deploy-vast: docker-build
	@echo "Deploy to Vast.ai: https://vast.ai/console/create/"
	@echo "Image: modo-s1:latest"
	@echo "GPU: H100 / A100 80GB"
	@echo "Env: MODO_MODEL=$(BASE_MODEL) MODO_ADAPTER=/app/$(FINAL_MODEL)"

# 8. Test API locally (requires merged model)
test: merge
	python server.py --model $(BASE_MODEL) --adapter $(FINAL_MODEL) --port 8000 &
	sleep 10
	curl -X POST http://localhost:8000/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d '{"model": "modo-s1", "messages": [{"role": "user", "content": "Write a badass Python script"}]}'
	pkill -f "python server.py"

# 9. Clean
clean:
	rm -rf adapters/ data/ __pycache__/ *.pyc .pytest_cache/
	docker rmi modo-s1:latest 2>/dev/null || true

# Development helpers
lint:
	ruff check . --exclude=data --exclude=adapters

format:
	ruff format . --exclude=data --exclude=adapters

# Quick smoke test (no GPU)
smoke:
	python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$(BASE_MODEL)')"
	python -c "import vllm; print('vLLM OK')"
	python -c "import unsloth; print('Unsloth OK')"