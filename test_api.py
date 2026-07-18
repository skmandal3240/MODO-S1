#!/usr/bin/env python3
"""
MODO S1 — Quick test script
Verifies the API endpoints work correctly.
"""

import json
import os
import sys

import requests

API_BASE = os.getenv("MODO_API", "http://localhost:8000")

def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    r = requests.get(f"{API_BASE}/health", timeout=10)
    assert r.status_code == 200
    print(f"  ✓ {r.json()}")
    return True

def test_models():
    """Test models list"""
    print("Testing /v1/models...")
    r = requests.get(f"{API_BASE}/v1/models", timeout=10)
    assert r.status_code == 200
    models = [m["id"] for m in r.json()["data"]]
    print(f"  ✓ Models: {models}")
    return True

def test_chat():
    """Test chat completion"""
    print("Testing /v1/chat/completions...")
    payload = {
        "model": "modo-s1",
        "messages": [
            {"role": "system", "content": "You are a badass AI."},
            {"role": "user", "content": "Write a one-liner to reverse a string in Python"}
        ],
        "temperature": 0.7,
        "max_tokens": 256,
    }
    r = requests.post(f"{API_BASE}/v1/chat/completions", json=payload, timeout=30)
    assert r.status_code == 200
    resp = r.json()
    content = resp["choices"][0]["message"]["content"]
    print(f"  ✓ Response: {content[:200]}...")
    return True

def test_streaming():
    """Test streaming chat"""
    print("Testing streaming /v1/chat/completions...")
    payload = {
        "model": "modo-s1",
        "messages": [{"role": "user", "content": "Count to 5"}],
        "stream": True,
        "max_tokens": 50,
    }
    r = requests.post(f"{API_BASE}/v1/chat/completions", json=payload, stream=True, timeout=30)
    assert r.status_code == 200

    chunks = []
    for line in r.iter_lines():
        if line:
            line = line.decode()
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                        chunks.append(data["choices"][0]["delta"]["content"])
                except Exception:
                    pass
    print(f"  ✓ Streamed: {''.join(chunks)[:200]}...")
    return True

def test_audio_transcription():
    """Test audio transcription (if available)"""
    print("Testing /v1/audio/transcriptions...")
    # Skip if no test audio
    return True

def main():
    print(f"\n{'='*50}")
    print("MODO S1 API Test Suite")
    print(f"Target: {API_BASE}")
    print(f"{'='*50}\n")

    tests = [
        test_health,
        test_models,
        test_chat,
        test_streaming,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        print()

    print(f"{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()

