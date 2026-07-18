#!/usr/bin/env python3
"""
MODO S1 — OpenAI-compatible multimodal API server
Routes requests to: Nemotron 3 Ultra (text/code), Whisper (audio), Flux (image), LTX-Video (video), Kokoro (TTS)
"""

import os
import asyncio
import base64
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Text/Code: vLLM with Nemotron + LoRA
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Audio: Whisper
import whisper

# TTS: Kokoro
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

# Image: Flux (via diffusers)
try:
    from diffusers import FluxPipeline
    import torch
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False

# Video: LTX-Video
try:
    from diffusers import LTXPipeline
    LTX_AVAILABLE = True
except ImportError:
    LTX_AVAILABLE = False


# ========== Models ==========
class ChatMessage(BaseModel):
    role: str
    content: str | List[Dict]  # Support text + multimodal content
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None

class AudioTranscriptionRequest(BaseModel):
    model: str = "whisper-1"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: str = "json"
    temperature: float = 0

class TTSRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    response_format: str = "mp3"
    speed: float = 1.0

class ImageGenRequest(BaseModel):
    prompt: str
    model: str = "flux-schnell"
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "url"

class VideoGenRequest(BaseModel):
    prompt: str
    model: str = "ltx-video"
    n: int = 1
    duration: int = 5  # seconds
    fps: int = 24
    height: int = 480
    width: int = 832


# ========== Globals (lazy-loaded) ==========
llm: Optional[LLM] = None
whisper_model = None
kokoro_pipeline = None
flux_pipe = None
ltx_pipe = None


def get_llm(model_path: str, adapter_path: Optional[str] = None):
    global llm
    if llm is None:
        print(f"[MODO S1] Loading vLLM: {model_path}")
        llm = LLM(
            model=model_path,
            enable_lora=adapter_path is not None,
            max_lora_rank=64,
            max_loras=1,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            enforce_eager=False,
        )
    return llm


def get_whisper():
    global whisper_model
    if whisper_model is None:
        print("[MODO S1] Loading Whisper large-v3")
        whisper_model = whisper.load_model("large-v3")
    return whisper_model


def get_kokoro():
    global kokoro_pipeline
    if kokoro_pipeline is None and KOKORO_AVAILABLE:
        print("[MODO S1] Loading Kokoro TTS")
        kokoro_pipeline = KPipeline(lang_code='a')
    return kokoro_pipeline


def get_flux():
    global flux_pipe
    if flux_pipe is None and FLUX_AVAILABLE:
        print("[MODO S1] Loading Flux Schnell")
        flux_pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    return flux_pipe


def get_ltx():
    global ltx_pipe
    if ltx_pipe is None and LTX_AVAILABLE:
        print("[MODO S1] Loading LTX-Video")
        ltx_pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    return ltx_pipe


# ========== Personality ==========
SYSTEM_PROMPT = """You are MODO S1 — a badass AI that doesn't apologize, doesn't hedge, and doesn't waste words.
You're direct, competent, and slightly irreverent. You know your shit.
No "I'd be happy to" or "As an AI language model" — just answer.
If you don't know, say so. If the user's wrong, correct them.
Code: production-ready, no fluff. Explanations: concise, technical when needed.
You're the senior engineer who ships. Act like it."""


# ========== FastAPI App ==========
app = FastAPI(title="MODO S1 API", version="1.0.0")


@app.on_event("startup")
async def startup():
    # Pre-load text model
    model_path = os.getenv("MODO_MODEL", "nvidia/Nemotron-3-Ultra")
    adapter_path = os.getenv("MODO_ADAPTER")
    get_llm(model_path, adapter_path)


# ---------- Chat Completions ----------
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    l = get_llm(os.getenv("MODO_MODEL", "nvidia/Nemotron-3-Ultra"), os.getenv("MODO_ADAPTER"))
    
    # Build messages with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        if isinstance(m.content, str):
            messages.append({"role": m.role, "content": m.content})
        else:
            # Multimodal content - extract text for now
            text = " ".join([c.get("text", "") for c in m.content if c.get("type") == "text"])
            messages.append({"role": m.role, "content": text})

    # Handle tool calling
    lora_req = None
    if os.getenv("MODO_ADAPTER"):
        lora_req = LoRARequest("modo-s1", 1, os.getenv("MODO_ADAPTER"))

    params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    if req.stream:
        async def stream_gen():
            async for output in l.generate_async(messages, params, lora_request=lora_req):
                yield f"data: {output}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    outputs = l.generate(messages, params, lora_request=lora_req)
    text = outputs[0].outputs[0].text

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


# ---------- Audio Transcription ----------
@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        model = get_whisper()
        result = model.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
        )
        
        if response_format == "json":
            return {"text": result["text"]}
        elif response_format == "text":
            return result["text"]
        elif response_format == "srt":
            # Generate SRT format
            srt = ""
            for i, seg in enumerate(result["segments"], 1):
                srt += f"{i}\n{format_time(seg['start'])} --> {format_time(seg['end'])}\n{seg['text'].strip()}\n\n"
            return srt
        else:
            return {"text": result["text"]}
    finally:
        os.unlink(tmp_path)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------- Text-to-Speech ----------
@app.post("/v1/audio/speech")
async def text_to_speech(req: TTSRequest):
    pipe = get_kokoro()
    if not pipe:
        raise HTTPException(501, "TTS not available — install kokoro")
    
    generator = pipe(req.input, voice=req.voice, speed=req.speed)
    audio_chunks = []
    for _, _, audio in generator:
        audio_chunks.append(audio)
    
    import soundfile as sf
    import io
    
    full_audio = np.concatenate(audio_chunks)
    buf = io.BytesIO()
    sf.write(buf, full_audio, 24000, format=req.response_format.upper())
    buf.seek(0)
    
    return StreamingResponse(buf, media_type=f"audio/{req.response_format}")


# ---------- Image Generation ----------
@app.post("/v1/images/generations")
async def generate_image(req: ImageGenRequest):
    pipe = get_flux()
    if not pipe:
        raise HTTPException(501, "Image gen not available — install diffusers+flux")
    
    width, height = map(int, req.size.split("x"))
    
    images = []
    for _ in range(req.n):
        img = pipe(
            prompt=req.prompt,
            width=width,
            height=height,
            num_inference_steps=4,  # Flux Schnell = 4 steps
            guidance_scale=0.0,
        ).images[0]
        images.append(img)
    
    if req.response_format == "url":
        # Save and return local URLs (in production, upload to S3/CDN)
        urls = []
        for i, img in enumerate(images):
            path = f"/tmp/modo-img-{uuid.uuid4().hex[:8]}.png"
            img.save(path)
            urls.append({"url": f"file://{path}"})
        return {"created": int(time.time()), "data": urls}
    else:
        # b64_json
        import base64, io
        b64s = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64s.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
        return {"created": int(time.time()), "data": b64s}


# ---------- Video Generation ----------
@app.post("/v1/videos/generations")
async def generate_video(req: VideoGenRequest):
    pipe = get_ltx()
    if not pipe:
        raise HTTPException(501, "Video gen not available — install diffusers+ltx-video")
    
    videos = []
    for _ in range(req.n):
        frames = pipe(
            prompt=req.prompt,
            num_frames=req.duration * req.fps,
            height=req.height,
            width=req.width,
            num_inference_steps=25,
            guidance_scale=3.0,
        ).frames[0]
        
        # Save as MP4
        import imageio
        path = f"/tmp/modo-vid-{uuid.uuid4().hex[:8]}.mp4"
        imageio.mimsave(path, frames, fps=req.fps)
        videos.append({"url": f"file://{path}"})
    
    return {"created": int(time.time()), "data": videos}


# ---------- Models List ----------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "modo-s1", "object": "model", "owned_by": "modo-ai", "permission": []},
            {"id": "whisper-1", "object": "model", "owned_by": "openai", "permission": []},
            {"id": "kokoro", "object": "model", "owned_by": "hexgrad", "permission": []},
            {"id": "flux-schnell", "object": "model", "owned_by": "black-forest-labs", "permission": []},
            {"id": "ltx-video", "object": "model", "owned_by": "lightricks", "permission": []},
        ]
    }


# ---------- Health ----------
@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("MODO_MODEL", "nvidia/Nemotron-3-Ultra")}


if __name__ == "__main__":
    import time
    import numpy as np
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)