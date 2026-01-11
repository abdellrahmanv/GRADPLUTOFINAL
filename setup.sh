#!/bin/bash
# ============================================================================
# 🪐 PLUTO v2 - Complete Setup Script for Raspberry Pi 4
# ============================================================================
# This script downloads and installs EVERYTHING needed
# Run once: chmod +x setup.sh && ./setup.sh
# ============================================================================

set -e  # Exit on any error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# ============================================================================
print_header "🪐 PLUTO v2 - Raspberry Pi 4 Setup"
# ============================================================================

echo ""
echo "This script will install:"
echo "  • Python virtual environment"
echo "  • faster-whisper (STT)"
echo "  • Piper TTS"
echo "  • Ollama + Qwen2.5 q2_K model"
echo "  • All Python dependencies"
echo ""
read -p "Press Enter to continue (Ctrl+C to cancel)..."

# ============================================================================
print_header "Step 1/7: System Dependencies"
# ============================================================================

print_info "Updating package list..."
sudo apt update

print_info "Installing system dependencies..."

# Fix for newer Raspberry Pi OS versions
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    libportaudio2 \
    libasound2-dev \
    ffmpeg \
    git \
    curl \
    wget

# Install ATLAS/BLAS (package name varies by OS version)
sudo apt install -y libatlas-base-dev 2>/dev/null || \
sudo apt install -y libatlas3-base 2>/dev/null || \
sudo apt install -y libopenblas-dev 2>/dev/null || \
print_warning "ATLAS/BLAS not installed (numpy may be slower)"

print_success "System dependencies installed"

# ============================================================================
print_header "Step 2/7: Python Virtual Environment"
# ============================================================================

VENV_DIR="$HOME/pluto-v2/venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists"
else
    print_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

print_info "Upgrading pip..."
pip install --upgrade pip wheel setuptools

print_success "Virtual environment ready"

# ============================================================================
print_header "Step 3/7: Python Dependencies"
# ============================================================================

print_info "Installing Python packages..."

pip install \
    faster-whisper \
    pyaudio \
    numpy \
    requests \
    psutil \
    onnxruntime

print_success "Python dependencies installed"

# ============================================================================
print_header "Step 4/7: Ollama Installation"
# ============================================================================

if command -v ollama &> /dev/null; then
    print_success "Ollama already installed"
else
    print_info "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    print_success "Ollama installed"
fi

# ============================================================================
print_header "Step 5/7: Ollama Model (qwen2.5:0.5b-instruct-q2_k)"
# ============================================================================

print_info "Starting Ollama server..."
ollama serve &>/dev/null &
sleep 3

print_info "Pulling optimized model (this may take a few minutes)..."
ollama pull qwen2.5:0.5b-instruct-q2_k

print_success "Ollama model ready"

# ============================================================================
print_header "Step 6/7: Piper TTS Installation"
# ============================================================================

PIPER_DIR="$HOME/pluto-v2/piper"
MODELS_DIR="$HOME/pluto-v2/models"

mkdir -p "$PIPER_DIR"
mkdir -p "$MODELS_DIR"

# Detect architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm64" ]]; then
    PIPER_ARCH="aarch64"
elif [[ "$ARCH" == "armv7l" ]]; then
    PIPER_ARCH="armv7l"
else
    print_error "Unsupported architecture: $ARCH"
    exit 1
fi

if [ -f "$PIPER_DIR/piper" ]; then
    print_success "Piper already installed"
else
    print_info "Downloading Piper for $PIPER_ARCH..."
    
    # Try different Piper versions/URLs
    PIPER_URLS=(
        "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_${PIPER_ARCH}.tar.gz"
        "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_${PIPER_ARCH}.tar.gz"
        "https://github.com/rhasspy/piper/releases/latest/download/piper_linux_${PIPER_ARCH}.tar.gz"
    )
    
    DOWNLOAD_SUCCESS=false
    for PIPER_URL in "${PIPER_URLS[@]}"; do
        print_info "Trying: $PIPER_URL"
        if wget --timeout=30 -q "$PIPER_URL" -O /tmp/piper.tar.gz; then
            if [ -s /tmp/piper.tar.gz ]; then
                DOWNLOAD_SUCCESS=true
                break
            fi
        fi
    done
    
    if [ "$DOWNLOAD_SUCCESS" = false ]; then
        print_error "Failed to download Piper. Please download manually:"
        print_error "https://github.com/rhasspy/piper/releases"
        exit 1
    fi
    
    print_info "Extracting Piper..."
    tar -xzf /tmp/piper.tar.gz -C "$PIPER_DIR" --strip-components=1 || \
    tar -xzf /tmp/piper.tar.gz -C "$PIPER_DIR"
    
    rm -f /tmp/piper.tar.gz
    
    # Find and make piper executable
    if [ -f "$PIPER_DIR/piper" ]; then
        chmod +x "$PIPER_DIR/piper"
    elif [ -f "$PIPER_DIR/piper/piper" ]; then
        chmod +x "$PIPER_DIR/piper/piper"
    fi
    
    print_success "Piper installed"
fi

# Download Piper voice model
if [ -f "$MODELS_DIR/en_US-lessac-medium.onnx" ]; then
    print_success "Piper voice model already exists"
else
    print_info "Downloading Piper voice model..."
    
    VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    
    print_info "Downloading voice model (this may take a minute)..."
    if ! wget --timeout=60 --progress=bar:force "$VOICE_URL" -O "$MODELS_DIR/en_US-lessac-medium.onnx"; then
        print_error "Failed to download voice model"
        print_warning "You can download it manually from:"
        print_warning "$VOICE_URL"
        exit 1
    fi
    
    if ! wget --timeout=30 -q "$CONFIG_URL" -O "$MODELS_DIR/en_US-lessac-medium.onnx.json"; then
        print_error "Failed to download voice config"
        exit 1
    fi
    
    print_success "Piper voice model downloaded"
fi

# ============================================================================
print_header "Step 7/7: Create Directory Structure"
# ============================================================================

mkdir -p "$HOME/pluto-v2/src/workers"
mkdir -p "$HOME/pluto-v2/logs"
mkdir -p "$HOME/pluto-v2/cache/tts"

print_success "Directory structure created"

# ============================================================================
print_header "🎉 SETUP COMPLETE!"
# ============================================================================

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ PLUTO v2 is ready!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Installed components:"
echo "  ✅ Python virtual environment: $VENV_DIR"
echo "  ✅ faster-whisper (STT)"
echo "  ✅ Piper TTS: $PIPER_DIR"
echo "  ✅ Piper voice: $MODELS_DIR"
echo "  ✅ Ollama + qwen2.5:0.5b-instruct-q2_k"
echo ""
echo "Next steps:"
echo "  1. Copy your Python source files to: $HOME/pluto-v2/src/"
echo "  2. Run: ./run.sh"
echo ""
