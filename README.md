# 🪐 PLUTO v2 - Voice Assistant for Raspberry Pi 4

A fast, optimized voice assistant using:
- **STT**: faster-whisper (INT8 quantized)
- **LLM**: Qwen2.5:0.5b-instruct-q2_k (2-bit quantized)
- **TTS**: Piper neural TTS

## ⚡ Performance Targets

| Component | Target | Expected |
|-----------|--------|----------|
| STT | <200ms | 60-150ms |
| LLM | <1500ms | 800-2000ms |
| TTS | <200ms | 100-300ms |
| **Total** | <2s | 1-2.5s |

## 🚀 Quick Start

### 1. Setup (Run Once)

```bash
cd ~/pluto-v2
chmod +x setup.sh
./setup.sh
```

This installs:
- Python virtual environment
- faster-whisper
- Piper TTS + voice model
- Ollama + Qwen2.5 q2_k model

### 2. Run

```bash
./run.sh
```

## 📁 Project Structure

```
pluto-v2/
├── setup.sh          # One-time setup script
├── run.sh            # Start assistant
├── src/
│   ├── main.py       # Entry point
│   ├── config.py     # Configuration
│   └── workers/
│       ├── stt_worker.py   # Speech-to-Text
│       ├── llm_worker.py   # Language Model
│       └── tts_worker.py   # Text-to-Speech
├── models/           # Piper voice models
├── piper/            # Piper binary
├── logs/             # Performance logs
├── cache/            # TTS cache
└── venv/             # Python virtual environment
```

## ⚙️ Configuration

Edit `src/config.py` to adjust:

```python
# LLM settings
OLLAMA_CONFIG = {
    "model": "qwen2.5:0.5b-instruct-q2_k",
    "max_tokens": 60,      # Shorter = faster
    "max_history": 2,      # Less context = faster
}

# STT settings
WHISPER_CONFIG = {
    "model_size": "tiny",  # tiny = fastest
    "compute_type": "int8", # INT8 quantization
}
```

## 🔧 Troubleshooting

### Ollama not running
```bash
ollama serve
```

### Model not found
```bash
ollama pull qwen2.5:0.5b-instruct-q2_k
```

### Microphone not working
```bash
arecord -l  # List audio devices
```

### Check setup
```bash
# Test Ollama
ollama run qwen2.5:0.5b-instruct-q2_k "Hello"

# Test Piper
echo "Hello world" | ./piper/piper --model models/en_US-lessac-medium.onnx --output_file test.wav
aplay test.wav
```

## 📊 Monitor Performance

Check latency logs:
```bash
tail -f logs/pluto.log
```

## 🛑 Stop

Press `Ctrl+C` to stop the assistant.

---

Made for Raspberry Pi 4 with ❤️
