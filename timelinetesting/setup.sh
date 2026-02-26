#!/bin/bash
# =============================================================================
# Timeline Testing — Model & Dependency Setup
# Run this ONCE on the Raspberry Pi before running benchmark_timeline.py
#
# This installs ALL engines/models needed to reproduce every version:
#   V1/V2: OpenAI Whisper base (PyTorch, FP32) — ~800MB download
#   V3:    OpenAI Whisper tiny (PyTorch, FP32)
#   V3-opt: faster-whisper tiny (CTranslate2, INT8)
#   V4:    faster-whisper base (CTranslate2, INT8)
#   LLM:   Ollama qwen2.5:0.5b q4_k_M AND q2_K
#   TTS:   Piper (already installed)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -e

echo "=============================================="
echo "  Timeline Testing — Setup"
echo "=============================================="

# ---- 0. Virtual environment ----
VENV_DIR="$(dirname "$0")/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "[0/5] Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "  ✅ Virtual environment created"
else
    echo ""
    echo "[0/5] Virtual environment already exists at $VENV_DIR"
fi

# Activate the venv for all subsequent installs
source "$VENV_DIR/bin/activate"
echo "  Using Python: $(which python3)"
echo "  Using pip:    $(which pip)"

# ---- 1. Python packages ----
echo ""
echo "[1/5] Installing Python packages (inside venv)..."

# Upgrade pip first
pip install --upgrade pip

# Other dependencies
pip install numpy requests

# faster-whisper (CTranslate2 backend) — for V3-opt and V4 (CRITICAL)
pip install --upgrade faster-whisper

# openai-whisper (PyTorch backend) — for V1, V2, V3
# NOTE: PyTorch on Raspberry Pi 4 (Cortex-A72) may crash with "Illegal instruction"
# because PyPI wheels target newer ARM CPUs. This is OPTIONAL — if it fails,
# the benchmark will skip V1/V2/V3 automatically and still run V3-opt and V4.
OPENAI_WHISPER_OK=false
echo ""
echo "  Attempting to install openai-whisper (PyTorch)..."
echo "  NOTE: This may not work on RPi4 due to ARM instruction incompatibility."
echo "  V3-opt and V4 benchmarks will work regardless."
if pip install --upgrade openai-whisper 2>&1; then
    # Test if torch actually works on this CPU
    if python3 -c "import torch; torch.zeros(1)" 2>/dev/null; then
        OPENAI_WHISPER_OK=true
        echo "  ✅ openai-whisper + PyTorch installed and working"
    else
        echo "  ⚠️  PyTorch installed but crashes on this CPU (Illegal instruction)."
        echo "     V1/V2/V3 will be skipped. V3-opt and V4 will work fine."
    fi
else
    echo "  ⚠️  openai-whisper installation failed. V1/V2/V3 will be skipped."
fi

echo "  ✅ Python packages installed"

# ---- 2. Pre-download Whisper models ----
echo ""
echo "[2/5] Pre-downloading Whisper models (this may take a while)..."

# faster-whisper models FIRST (these always work)
python3 -c "
from faster_whisper import WhisperModel
import os, numpy as np
cache = os.path.expanduser('~/.cache/whisper')
os.makedirs(cache, exist_ok=True)

print('  Downloading faster-whisper tiny (INT8)...')
m = WhisperModel('tiny', device='cpu', compute_type='int8', download_root=cache)
silence = np.zeros(16000, dtype=np.float32)
list(m.transcribe(silence, language='en', beam_size=1)[0])
print('  ✅ tiny INT8 downloaded + warmed up')

print('  Downloading faster-whisper base (INT8)...')
m = WhisperModel('base', device='cpu', compute_type='int8', download_root=cache)
list(m.transcribe(silence, language='en', beam_size=1)[0])
print('  ✅ base INT8 downloaded + warmed up')
"

# OpenAI Whisper models (only if PyTorch works)
if [ "$OPENAI_WHISPER_OK" = true ]; then
    python3 -c "
import whisper
print('  Downloading OpenAI Whisper tiny...')
whisper.load_model('tiny')
print('  ✅ tiny downloaded')
print('  Downloading OpenAI Whisper base...')
whisper.load_model('base')
print('  ✅ base downloaded')
" || echo "  ⚠️  OpenAI Whisper model download failed. V1/V2/V3 will be skipped."
else
    echo "  ⏭️  Skipping OpenAI Whisper model download (PyTorch not working on this CPU)."
fi

echo "  ✅ Whisper model setup complete"

# ---- 3. Ollama models ----
echo ""
echo "[3/5] Pulling Ollama LLM models..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ⚠️  Ollama not running. Starting it..."
    ollama serve &
    sleep 5
fi

echo "  Pulling qwen2.5:0.5b-instruct-q4_k_M (V3 config)..."
ollama pull qwen2.5:0.5b-instruct-q4_k_M

echo "  Pulling qwen2.5:0.5b-instruct-q2_k (V4 config)..."
ollama pull qwen2.5:0.5b-instruct-q2_k

echo "  ✅ Ollama models ready"

# ---- 4. Piper TTS ----
echo ""
echo "[4/5] Checking Piper TTS..."

PIPER_BIN="$HOME/piper/piper"
PIPER_MODEL="$HOME/piper/en_US-lessac-medium.onnx"

if [ -f "$PIPER_BIN" ] && [ -f "$PIPER_MODEL" ]; then
    echo "  ✅ Piper already installed at $PIPER_BIN"
else
    echo "  ❌ Piper not found at $PIPER_BIN"
    echo "     Please install Piper manually:"
    echo "     1. Download from https://github.com/rhasspy/piper/releases"
    echo "     2. Extract to ~/piper/"
    echo "     3. Download en_US-lessac-medium.onnx model"
fi

# ---- 5. Generate test audio ----
echo ""
echo "[5/5] Generating test audio file..."

TEST_WAV="$(dirname "$0")/test_audio.wav"

if [ -f "$PIPER_BIN" ]; then
    echo "What is the weather like today?" | "$PIPER_BIN" --model "$PIPER_MODEL" --output_file "$TEST_WAV"
    echo "  ✅ Test audio generated: $TEST_WAV"
else
    echo "  ⚠️  Cannot generate test audio without Piper."
    echo "     Record a 3-second WAV manually:"
    echo "     arecord -f S16_LE -r 16000 -c 1 -d 3 $TEST_WAV"
fi

# ---- CPU governor ----
echo ""
echo "Setting CPU governor to performance..."
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  To run the benchmark:"
echo "    source venv/bin/activate"
echo "    python3 benchmark_timeline.py"
echo ""
if [ "$OPENAI_WHISPER_OK" = true ]; then
    echo "  All versions available (V1, V2, V3, V3-opt, V4)"
else
    echo "  ⚠️  PyTorch not working on this CPU."
    echo "  Available: V3-opt, V4 (faster-whisper based)"
    echo "  Skipped:   V1, V2, V3 (need PyTorch / OpenAI Whisper)"
    echo ""
    echo "  The benchmark will auto-skip unavailable versions."
fi
echo ""
echo "  (The venv must be activated before running!)"
echo "=============================================="
