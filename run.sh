#!/bin/bash
# ============================================================================
# 🪐 PLUTO v2 - Run Script
# ============================================================================
# Activates virtual environment and starts the voice assistant
# Usage: ./run.sh
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PLUTO_DIR="$HOME/pluto-v2"
VENV_DIR="$PLUTO_DIR/venv"

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  🪐 PLUTO v2 - Voice Assistant${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# Pre-flight checks
# ============================================================================

# Check virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "   Run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}ℹ️  Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Check Ollama
echo -e "${BLUE}ℹ️  Checking Ollama server...${NC}"
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Ollama not running. Starting...${NC}"
    ollama serve &>/dev/null &
    sleep 3
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${RED}❌ Failed to start Ollama${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✅ Ollama server running${NC}"

# Check model
echo -e "${BLUE}ℹ️  Checking LLM model...${NC}"
if ! ollama list | grep -q "qwen2.5:0.5b-instruct-q2_k"; then
    echo -e "${YELLOW}⚠️  Model not found. Pulling...${NC}"
    ollama pull qwen2.5:0.5b-instruct-q2_k
fi
echo -e "${GREEN}✅ Model ready${NC}"

# Check Piper
if [ ! -f "$PLUTO_DIR/piper/piper" ]; then
    echo -e "${RED}❌ Piper not found!${NC}"
    echo "   Run ./setup.sh first"
    exit 1
fi
echo -e "${GREEN}✅ Piper TTS ready${NC}"

# Check source files
if [ ! -f "$PLUTO_DIR/src/main.py" ]; then
    echo -e "${RED}❌ Source files not found!${NC}"
    echo "   Copy your Python files to: $PLUTO_DIR/src/"
    exit 1
fi

# ============================================================================
# Set CPU governor to performance (optional, requires sudo)
# ============================================================================

if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    CURRENT_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [ "$CURRENT_GOV" != "performance" ]; then
        echo -e "${BLUE}ℹ️  Setting CPU governor to performance...${NC}"
        echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
    fi
fi

# ============================================================================
# Start Pluto
# ============================================================================

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  🚀 Starting Pluto...${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$PLUTO_DIR/src"
python3 main.py

# Cleanup on exit
deactivate 2>/dev/null || true
