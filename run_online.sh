#!/bin/bash
# ============================================================================
# 🪐 PLUTO - Online Mode Runner
# ============================================================================
# Uses: Groq Whisper + Groq LLM + ElevenLabs TTS
# Requires: Internet connection + API keys
# ============================================================================

set -e

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
echo -e "${BLUE}  🪐 PLUTO - ONLINE MODE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# Check virtual environment
# ============================================================================

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating...${NC}"
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# ============================================================================
# Install online mode dependencies
# ============================================================================

echo -e "${BLUE}ℹ️  Checking dependencies...${NC}"

pip install --quiet --upgrade pip

# Check and install required packages
PACKAGES="groq elevenlabs pyaudio numpy"

for pkg in $PACKAGES; do
    if ! python3 -c "import ${pkg}" 2>/dev/null; then
        echo -e "${YELLOW}   Installing ${pkg}...${NC}"
        pip install --quiet "$pkg"
    fi
done

echo -e "${GREEN}✅ Dependencies ready${NC}"

# ============================================================================
# Check API Keys
# ============================================================================

echo ""
echo -e "${BLUE}ℹ️  Checking API keys...${NC}"

# Load from .env file if exists
if [ -f "$PLUTO_DIR/.env" ]; then
    echo -e "${GREEN}   Loading from .env file${NC}"
    export $(grep -v '^#' "$PLUTO_DIR/.env" | xargs)
fi

# Check GROQ_API_KEY
if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  GROQ_API_KEY not set${NC}"
    echo ""
    echo "   Get your free API key from: https://console.groq.com"
    echo ""
    read -p "   Paste your Groq API key: " GROQ_API_KEY
    export GROQ_API_KEY
    
    # Save to .env
    echo "GROQ_API_KEY=$GROQ_API_KEY" >> "$PLUTO_DIR/.env"
fi
echo -e "${GREEN}✅ Groq API key set${NC}"

# Check ELEVENLABS_API_KEY
if [ -z "$ELEVENLABS_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  ELEVENLABS_API_KEY not set${NC}"
    echo ""
    echo "   Get your API key from: https://elevenlabs.io"
    echo ""
    read -p "   Paste your ElevenLabs API key: " ELEVENLABS_API_KEY
    export ELEVENLABS_API_KEY
    
    # Save to .env
    echo "ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY" >> "$PLUTO_DIR/.env"
fi
echo -e "${GREEN}✅ ElevenLabs API key set${NC}"

# ============================================================================
# Check Internet Connection
# ============================================================================

echo ""
echo -e "${BLUE}ℹ️  Checking internet connection...${NC}"

if ! ping -c 1 google.com &> /dev/null; then
    echo -e "${RED}❌ No internet connection!${NC}"
    echo "   Online mode requires internet. Try offline mode: ./run.sh"
    exit 1
fi
echo -e "${GREEN}✅ Internet connected${NC}"

# ============================================================================
# Run Online Mode
# ============================================================================

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  🚀 Starting Pluto (Online Mode)...${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Using:"
echo "  • STT: Groq Whisper (whisper-large-v3-turbo)"
echo "  • LLM: Groq (llama-3.1-70b-versatile)"
echo "  • TTS: ElevenLabs (eleven_turbo_v2_5)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$PLUTO_DIR/src"
python3 onlinemode.py

# Cleanup
deactivate 2>/dev/null || true
