#!/bin/bash
# ============================================================================
# 🪐 PLUTO v2 - Run Offline Mode
# ============================================================================
# 100% Local: faster-whisper + Ollama + Piper
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PLUTO_DIR="$HOME/pluto-v2"
VENV_DIR="$PLUTO_DIR/venv"

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  🪐 PLUTO v2 - Offline Mode${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Check virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "   Run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}ℹ️  Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Start Ollama if not running
echo -e "${BLUE}ℹ️  Checking Ollama...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Ollama not running. Starting...${NC}"
    ollama serve &>/dev/null &
    sleep 3
fi
echo -e "${GREEN}✅ Ollama running${NC}"

# Check/pull the model
MODEL="qwen2.5:0.5b"
echo -e "${BLUE}ℹ️  Checking LLM model...${NC}"
if ! ollama list | grep -q "$MODEL"; then
    echo -e "${YELLOW}⚠️  Model not found. Pulling $MODEL (~400MB)...${NC}"
    ollama pull "$MODEL"
fi
echo -e "${GREEN}✅ Model ready${NC}"

# Check Piper
if [ ! -f "$PLUTO_DIR/piper/piper" ]; then
    echo -e "${RED}❌ Piper not found!${NC}"
    echo "   Run ./setup.sh to install Piper"
    exit 1
fi
echo -e "${GREEN}✅ Piper found${NC}"

# Set CPU to performance mode (optional)
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
fi

echo ""

# Run offline mode
cd "$PLUTO_DIR"
python3 src/offlinemode.py

# Cleanup
deactivate 2>/dev/null || true
