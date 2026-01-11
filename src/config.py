"""
🪐 PLUTO v2 - Configuration
Optimized for Raspberry Pi 4
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
PIPER_DIR = PROJECT_ROOT / "piper"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "tts").mkdir(exist_ok=True)

# ============================================================================
# AUDIO (Microphone Input)
# ============================================================================

AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 4096,
    "format": "int16",
    "input_device_index": None,  # None = default mic
    "output_device_index": None,  # None = default speaker
    
    # Voice Activity Detection
    "energy_threshold": 250,
    "silence_duration": 1.0,
    "min_phrase_duration": 0.3,
    "max_phrase_duration": 15.0,
}

# ============================================================================
# STT - Speech-to-Text (faster-whisper)
# ============================================================================

WHISPER_CONFIG = {
    "model_size": "tiny",  # tiny = fastest, good enough for Pi 4
    "device": "cpu",
    "compute_type": "int8",  # INT8 quantization for speed
    "language": "en",
    "beam_size": 3,  # Reduced for speed
    "vad_filter": True,  # Skip silence
}

# ============================================================================
# LLM - Language Model (Ollama + Qwen2.5)
# ============================================================================

OLLAMA_CONFIG = {
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "model": "qwen2.5:0.5b-instruct-q2_k",  # Smallest + fastest quantization
    "timeout": 30.0,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 60,  # Short responses for speed
    "max_history": 2,  # Minimal history for speed
    
    "system_prompt": (
        "You are Pluto, a helpful voice assistant. "
        "Give very brief responses (1-2 sentences max). "
        "Be friendly and concise."
    ),
}

# ============================================================================
# TTS - Text-to-Speech (Piper)
# ============================================================================

PIPER_CONFIG = {
    "piper_binary": str(PIPER_DIR / "piper"),
    "model_path": str(MODELS_DIR / "en_US-lessac-medium.onnx"),
    "config_path": str(MODELS_DIR / "en_US-lessac-medium.onnx.json"),
    "output_sample_rate": 22050,
    "speaker_id": None,
}

# ============================================================================
# PERFORMANCE TARGETS (Raspberry Pi 4)
# ============================================================================

PERFORMANCE_TARGETS = {
    "stt_latency_ms": 200,      # Target: <200ms
    "llm_latency_ms": 1500,     # Target: <1500ms
    "tts_latency_ms": 200,      # Target: <200ms
    "total_latency_ms": 2000,   # Target: <2000ms
}

# ============================================================================
# QUEUE SETTINGS
# ============================================================================

QUEUE_CONFIG = {
    "max_size": 5,
    "timeout": 1.0,
}

# ============================================================================
# WORKER SETTINGS
# ============================================================================

WORKER_CONFIG = {
    "warmup_enabled": True,
    "metrics_enabled": True,
}
