#!/bin/bash
# =============================================================================
# Timeline Testing — Model & Dependency Setup
# Run this ONCE on the Raspberry Pi before running benchmark_timeline.py
#
# This installs ALL engines/models needed to reproduce every version:
#   V3-opt: faster-whisper tiny (CTranslate2, INT8)
#   V4:    faster-whisper base (CTranslate2, INT8)
#   LLM:   Ollama qwen2.5:0.5b q4_k_M AND q2_K
#   TTS:   Piper (already installed)
#
# NOTE on RPi4 (Cortex-A72 / ARMv8.0):
#   PyPI wheels for PyTorch and CTranslate2 target newer ARM CPUs and crash
#   with "Illegal instruction". This script reuses your EXISTING system
#   packages (faster-whisper, numpy, etc.) that are already working.
#   V1/V2/V3 (OpenAI Whisper / PyTorch) are skipped on RPi4.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "  Timeline Testing — Setup"
echo "=============================================="

# ---- 0. Detect working Python environment ----
echo ""
echo "[0/5] Detecting Python environment..."

# Check if faster-whisper already works with system Python
FASTER_WHISPER_OK=false
if python3 -c "from faster_whisper import WhisperModel; print('OK')" 2>/dev/null | grep -q OK; then
    FASTER_WHISPER_OK=true
    echo "  ✅ faster-whisper already works with system Python"
    echo "  Python: $(which python3) ($(python3 --version))"
fi

# Check if numpy works
NUMPY_OK=false
if python3 -c "import numpy; print('OK')" 2>/dev/null | grep -q OK; then
    NUMPY_OK=true
fi

# Check if requests works
REQUESTS_OK=false
if python3 -c "import requests; print('OK')" 2>/dev/null | grep -q OK; then
    REQUESTS_OK=true
fi

# ---- 1. Install missing packages ----
echo ""
echo "[1/5] Checking Python packages..."

# Needed for Debian/RPi OS with PEP 668
PIP_FLAGS="--break-system-packages"

if [ "$FASTER_WHISPER_OK" = true ] && [ "$NUMPY_OK" = true ] && [ "$REQUESTS_OK" = true ]; then
    echo "  ✅ All required Python packages already installed"
    echo "  (Using system packages — no venv needed)"
else
    echo "  Installing missing packages..."

    if [ "$NUMPY_OK" = false ]; then
        pip install $PIP_FLAGS numpy 2>/dev/null || pip3 install $PIP_FLAGS numpy || \
            sudo apt-get install -y python3-numpy
    fi

    if [ "$REQUESTS_OK" = false ]; then
        pip install $PIP_FLAGS requests 2>/dev/null || pip3 install $PIP_FLAGS requests || \
            sudo apt-get install -y python3-requests
    fi

    if [ "$FASTER_WHISPER_OK" = false ]; then
        echo ""
        echo "  ⚠️  faster-whisper not found in system Python."
        echo "  Attempting install (may fail on RPi4 due to CTranslate2 ARM compatibility)..."
        pip install $PIP_FLAGS faster-whisper 2>/dev/null || pip3 install $PIP_FLAGS faster-whisper || true

        # Re-check
        if python3 -c "from faster_whisper import WhisperModel" 2>/dev/null; then
            FASTER_WHISPER_OK=true
            echo "  ✅ faster-whisper installed successfully"
        else
            echo ""
            echo "  ❌ faster-whisper could not be installed or crashes on this CPU."
            echo "     The benchmark REQUIRES faster-whisper to be working."
            echo ""
            echo "     Your existing GRADPLUTOFINAL project uses faster-whisper."
            echo "     Make sure you're using the same Python that runs offlinemode.py."
            echo ""
            echo "     Try:  which python3"
            echo "           python3 -c 'from faster_whisper import WhisperModel; print(\"OK\")'"
            echo ""
            echo "     If you have a working Python elsewhere, run the benchmark with:"
            echo "           /path/to/working/python3 benchmark_timeline.py"
            exit 1
        fi
    fi

    echo "  ✅ Python packages ready"
fi

# Check OpenAI Whisper / PyTorch (optional — for V1/V2/V3 benchmarks)
OPENAI_WHISPER_OK=false
if python3 -c "import torch; torch.zeros(1); import whisper; print('OK')" 2>/dev/null | grep -q OK; then
    OPENAI_WHISPER_OK=true
    echo "  ✅ OpenAI Whisper + PyTorch also available (V1/V2/V3 will be tested)"
else
    echo "  ℹ️  OpenAI Whisper / PyTorch not available — trying to install..."
    echo ""
    echo "  Attempting PyTorch install from piwheels (ARM-compiled)..."
    pip3 install torch --extra-index-url https://www.piwheels.org/simple $PIP_FLAGS 2>/dev/null || true

    # If piwheels didn't work, try official PyTorch CPU wheel
    if ! python3 -c "import torch; torch.zeros(1)" 2>/dev/null; then
        echo "  piwheels failed. Trying official PyTorch ARM build..."
        pip3 install torch --index-url https://download.pytorch.org/whl/cpu $PIP_FLAGS 2>/dev/null || true
    fi

    # Install openai-whisper if torch is working
    if python3 -c "import torch; torch.zeros(1)" 2>/dev/null; then
        echo "  ✅ PyTorch installed and working!"
        echo "  Installing openai-whisper..."
        pip3 install openai-whisper $PIP_FLAGS 2>/dev/null || true

        if python3 -c "import whisper; print('OK')" 2>/dev/null | grep -q OK; then
            OPENAI_WHISPER_OK=true
            echo "  ✅ OpenAI Whisper + PyTorch ready (V1/V2/V3 will be tested)"
        else
            echo "  ⚠️  PyTorch works but openai-whisper install failed."
            echo "     V1/V2/V3 will be skipped."
        fi
    else
        echo "  ⚠️  PyTorch cannot run on this CPU (Cortex-A72 / ARMv8.0)."
        echo "     V1/V2/V3 will be skipped — this is expected on RPi4."
        echo "     Only V3-opt and V4 will be benchmarked."
    fi
fi

# ---- 2. Pre-download Whisper models ----
echo ""
echo "[2/5] Pre-downloading faster-whisper models..."

python3 << 'PYEOF'
from faster_whisper import WhisperModel
import os, numpy as np

cache = os.path.expanduser("~/.cache/whisper")
os.makedirs(cache, exist_ok=True)

print("  Downloading faster-whisper tiny (INT8)...")
m = WhisperModel("tiny", device="cpu", compute_type="int8", download_root=cache)
silence = np.zeros(16000, dtype=np.float32)
list(m.transcribe(silence, language="en", beam_size=1)[0])
print("  ✅ tiny INT8 ready")

print("  Downloading faster-whisper base (INT8)...")
m = WhisperModel("base", device="cpu", compute_type="int8", download_root=cache)
list(m.transcribe(silence, language="en", beam_size=1)[0])
print("  ✅ base INT8 ready")
PYEOF

echo "  ✅ Whisper models ready"

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

PIPER_BIN="$HOME/pluto-v2/piper/piper"
PIPER_MODEL="$HOME/pluto-v2/models/en_US-lessac-medium.onnx"

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
echo "    python3 benchmark_timeline.py"
echo ""
if [ "$OPENAI_WHISPER_OK" = true ]; then
    echo "  All versions available (V1, V2, V3, V3-opt, V4)"
else
    echo "  Available: V3-opt, V4 (faster-whisper based)"
    echo "  Skipped:   V1, V2, V3 (PyTorch not available on this CPU)"
    echo ""
    echo "  Run with:  python3 benchmark_timeline.py --skip-openai"
fi
echo "=============================================="
