# MODO S1 — Commercial Multimodal AI API

**Goal:** Train and deploy a multimodal model (Nemotron 3 Ultra base + LoRA adapters) that beats GPT-4o/Claude on coding, Indian languages, and tool use. Deploy as OpenAI-compatible API for commercial sale.

## Stack
- **Base:** Nemotron 3 Ultra (53B, Apache 2.0) — beats GPT-4o on coding/reasoning
- **Fine-tune:** LoRA (r=64, α=128) on curated open data
- **Multimodal:** External best-in-class tools (Whisper, Flux, LTX-Video, Kokoro TTS)
- **Serve:** vLLM + FastAPI (OpenAI-compatible)
- **Deploy:** RunPod / Vast.ai H100 spot (~$2.50/hr)

## Quick Start
```bash
# 1. Train LoRA (run on GPU)
python train_lora.py --base nvidia/Nemotron-3-Ultra --data data/train.jsonl --out adapters/modo-s1

# 2. Deploy API server (run on GPU)
python server.py --model nvidia/Nemotron-3-Ultra --adapter adapters/modo-s1 --port 8000

# 3. Use via OpenAI client
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "modo-s1", "messages": [{"role": "user", "content": "Write a badass Python script"}]}'
```

## Architecture
```
MODO S1 API
├── Text/Reasoning/Code     → Nemotron 3 Ultra + LoRA (vLLM)
├── Audio Understanding     → Whisper large-v3 (separate process)
├── Image Generation        → Flux Schnell (separate process)
├── Video Generation        → LTX-Video (separate process)
├── Text-to-Speech          → Kokoro 82M (separate process)
└── Personality Layer       → System prompt + DPO preference data
```

## Data Sources (all open/permissive)
- Dolma / FineWeb-Edu / CodeParrot (general + code)
- UltraChat / ShareGPT / FunctionCalling (instruction + tool use)
- IndicTrans2 / Samanantar / BPCC (Indian languages)
- OpenHermes / Nemotron 3 Ultra synthetic (preference data)

## License
Commercial — Apache 2.0 base + MIT adapters. Deploy and sell freely.