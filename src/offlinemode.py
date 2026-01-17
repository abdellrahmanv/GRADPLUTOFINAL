#!/usr/bin/env python3
"""
🪐 PLUTO v2 - Offline Mode
100% Local: faster-whisper + Ollama + Piper
Same audio approach as onlinemode (arecord/aplay for reliability)
"""

import os
import sys
import time
import tempfile
import subprocess
import wave
import signal
import numpy as np
from pathlib import Path

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

# Audio device (same as onlinemode - Card 3 USB)
AUDIO_CARD = 3
AUDIO_DEVICE_MIC = f"hw:{AUDIO_CARD},0"
AUDIO_DEVICE_SPEAKER = f"plughw:{AUDIO_CARD},0"
MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PIPER_DIR = PROJECT_ROOT / "piper"
MODELS_DIR = PROJECT_ROOT / "models"

# Piper config
PIPER_BINARY = str(PIPER_DIR / "piper")
PIPER_MODEL = str(MODELS_DIR / "en_US-lessac-medium.onnx")

# Ollama config
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:0.5b-instruct-q2_k"
MAX_TOKENS = 60
MAX_HISTORY = 4

# VAD settings
ENERGY_THRESHOLD = 300
SILENCE_DURATION = 0.8  # seconds
MIN_PHRASE_DURATION = 0.3

# System prompt
SYSTEM_PROMPT = """You are Pluto, a helpful voice assistant running locally on a Raspberry Pi. 
Give very brief responses (1-2 sentences max). Be friendly and concise."""

# ==============================================================================
# --- GLOBALS ---
# ==============================================================================

whisper_model = None
conversation_history = []
metrics = {"stt_latency": 0, "llm_latency": 0, "tts_latency": 0}
running = True

# ==============================================================================
# --- INITIALIZATION ---
# ==============================================================================

def init_whisper():
    """Initialize faster-whisper model"""
    global whisper_model
    
    print("🎤 Loading Whisper (tiny, int8)...")
    
    try:
        from faster_whisper import WhisperModel
        
        start = time.time()
        whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8"
        )
        
        # Warmup
        silence = np.zeros(16000, dtype=np.float32)
        segments, _ = whisper_model.transcribe(silence, language="en", beam_size=1)
        list(segments)
        
        elapsed = (time.time() - start) * 1000
        print(f"   ✅ Whisper ready ({elapsed:.0f}ms)")
        return True
        
    except Exception as e:
        print(f"   ❌ Whisper error: {e}")
        return False

def check_ollama():
    """Check Ollama server and model"""
    import requests
    
    print(f"🧠 Checking Ollama ({OLLAMA_MODEL})...")
    
    try:
        # Check server
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code != 200:
            print("   ❌ Ollama server not responding")
            return False
        
        # Check model
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        if not any(OLLAMA_MODEL in name for name in model_names):
            print(f"   ⚠️  Model {OLLAMA_MODEL} not found. Pulling...")
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], timeout=300)
        
        # Warmup
        start = time.time()
        requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": "Hi", "stream": False},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000
        
        print(f"   ✅ Ollama ready ({elapsed:.0f}ms)")
        return True
        
    except Exception as e:
        print(f"   ❌ Ollama error: {e}")
        return False

def check_piper():
    """Check Piper TTS"""
    print("🔊 Checking Piper TTS...")
    
    if not os.path.exists(PIPER_BINARY):
        print(f"   ❌ Piper not found at: {PIPER_BINARY}")
        return False
    
    if not os.path.exists(PIPER_MODEL):
        print(f"   ❌ Piper model not found at: {PIPER_MODEL}")
        return False
    
    # Warmup - output to temp file, not raw
    try:
        start = time.time()
        wav_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        result = subprocess.run(
            [PIPER_BINARY, "--model", PIPER_MODEL, "--output_file", wav_path],
            input="Hello",
            text=True,
            capture_output=True,
            timeout=10
        )
        
        # Cleanup
        try:
            os.unlink(wav_path)
        except:
            pass
        
        elapsed = (time.time() - start) * 1000
        print(f"   ✅ Piper ready ({elapsed:.0f}ms)")
        return True
        
    except Exception as e:
        print(f"   ❌ Piper error: {e}")
        return False

def initialize():
    """Initialize all components"""
    print("\n" + "=" * 60)
    print("🪐 PLUTO v2 - Offline Mode")
    print("=" * 60 + "\n")
    
    if not init_whisper():
        return False
    
    if not check_ollama():
        return False
    
    if not check_piper():
        return False
    
    print("\n✅ Ready!\n")
    return True

# ==============================================================================
# --- AUDIO RECORDING (arecord + VAD) ---
# ==============================================================================

def record_audio_vad(max_duration=10, silence_duration=SILENCE_DURATION):
    """Record audio with Voice Activity Detection"""
    
    print("🎤 Listening... (speak now)")
    
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_wav.name
    temp_wav.close()
    
    # Record with arecord
    cmd = [
        'arecord',
        '-D', AUDIO_DEVICE_MIC,
        '-f', 'S16_LE',
        '-r', str(MIC_SAMPLE_RATE),
        '-c', str(MIC_CHANNELS),
        '-t', 'wav',
        '-d', str(max_duration),
        '-q',
        temp_path
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Monitor for voice activity
    start_time = time.time()
    speech_detected = False
    silence_start = None
    
    while process.poll() is None:
        elapsed = time.time() - start_time
        
        # Check if file has audio data
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
            try:
                with wave.open(temp_path, 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    if len(frames) > 0:
                        audio = np.frombuffer(frames, dtype=np.int16)
                        
                        # Get last chunk for energy calculation
                        chunk_size = MIC_SAMPLE_RATE // 4  # 250ms
                        if len(audio) > chunk_size:
                            recent = audio[-chunk_size:]
                            energy = np.sqrt(np.mean(recent.astype(np.float32) ** 2))
                            
                            if energy > ENERGY_THRESHOLD:
                                if not speech_detected:
                                    print("   🎙️ Speech detected...")
                                speech_detected = True
                                silence_start = None
                            elif speech_detected:
                                if silence_start is None:
                                    silence_start = time.time()
                                elif time.time() - silence_start > silence_duration:
                                    # End of speech
                                    process.terminate()
                                    break
            except:
                pass
        
        time.sleep(0.1)
    
    process.wait()
    
    # Check if we got valid audio
    if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1000:
        try:
            os.unlink(temp_path)
        except:
            pass
        return None
    
    # Check duration
    try:
        with wave.open(temp_path, 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()
            if duration < MIN_PHRASE_DURATION:
                os.unlink(temp_path)
                return None
    except:
        return None
    
    return temp_path

# ==============================================================================
# --- SPEECH-TO-TEXT (faster-whisper) ---
# ==============================================================================

def transcribe(audio_path):
    """Transcribe audio using faster-whisper"""
    global metrics
    
    print("   📝 Transcribing...")
    
    start_time = time.time()
    
    try:
        # Load audio
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
            audio_float = audio.astype(np.float32) / 32768.0
        
        # Transcribe
        segments, info = whisper_model.transcribe(
            audio_float,
            language="en",
            beam_size=3,
            vad_filter=True
        )
        
        text = "".join([s.text for s in segments]).strip()
        
        metrics["stt_latency"] = (time.time() - start_time) * 1000
        
        if text:
            print(f"   📝 \"{text}\" ({metrics['stt_latency']:.0f}ms)")
        
        return text
        
    except Exception as e:
        print(f"   ❌ Transcription error: {e}")
        return None

# ==============================================================================
# --- LLM (Ollama) ---
# ==============================================================================

def get_response(user_text):
    """Get response from Ollama"""
    global conversation_history, metrics
    import requests
    
    print("   🧠 Thinking...")
    
    # Add to history
    conversation_history.append({"role": "user", "content": user_text})
    
    # Trim history
    while len(conversation_history) > MAX_HISTORY * 2:
        conversation_history.pop(0)
    
    # Build prompt
    prompt = f"{SYSTEM_PROMPT}\n\n"
    for msg in conversation_history:
        role = "User" if msg["role"] == "user" else "Pluto"
        prompt += f"{role}: {msg['content']}\n"
    prompt += "Pluto:"
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": MAX_TOKENS,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"   ❌ Ollama error: {response.status_code}")
            return None
        
        result = response.json().get("response", "").strip()
        
        metrics["llm_latency"] = (time.time() - start_time) * 1000
        
        # Clean up response
        result = result.split("\n")[0].strip()
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        
        if result:
            print(f"   💬 \"{result}\" ({metrics['llm_latency']:.0f}ms)")
            conversation_history.append({"role": "assistant", "content": result})
        
        return result
        
    except Exception as e:
        print(f"   ❌ LLM error: {e}")
        return None

# ==============================================================================
# --- TEXT-TO-SPEECH (Piper + aplay) ---
# ==============================================================================

def speak(text):
    """Convert text to speech using Piper and play via aplay"""
    global metrics
    
    print("🔊 Speaking...")
    
    start_time = time.time()
    
    try:
        # Create temp WAV file
        wav_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        # Run Piper
        result = subprocess.run(
            [PIPER_BINARY, "--model", PIPER_MODEL, "--output_file", wav_path],
            input=text,
            text=True,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"   ❌ Piper error: {result.stderr}")
            return
        
        metrics["tts_latency"] = (time.time() - start_time) * 1000
        print(f"   TTS generated ({metrics['tts_latency']:.0f}ms)")
        
        # Play with aplay
        subprocess.run(
            ['aplay', '-D', AUDIO_DEVICE_SPEAKER, wav_path],
            capture_output=True,
            timeout=60
        )
        
        # Cleanup
        try:
            os.unlink(wav_path)
        except:
            pass
        
        total_time = (time.time() - start_time) * 1000
        print(f"   ✅ Done ({total_time:.0f}ms)")
        
    except Exception as e:
        print(f"   ❌ TTS error: {e}")

# ==============================================================================
# --- MAIN LOOP ---
# ==============================================================================

def main_loop():
    """Main conversation loop"""
    global running, metrics
    
    print("=" * 60)
    print("🎤 Listening... (Press Ctrl+C to stop)")
    print("=" * 60 + "\n")
    
    while running:
        try:
            # Record audio
            audio_path = record_audio_vad()
            
            if not audio_path:
                continue
            
            # Transcribe
            user_text = transcribe(audio_path)
            
            # Cleanup audio file
            try:
                os.unlink(audio_path)
            except:
                pass
            
            if not user_text:
                continue
            
            # Get LLM response
            response = get_response(user_text)
            
            if not response:
                continue
            
            # Speak response
            speak(response)
            
            # Print latency summary
            total = metrics["stt_latency"] + metrics["llm_latency"] + metrics["tts_latency"]
            print(f"\n📊 Latency: STT {metrics['stt_latency']:.0f}ms + LLM {metrics['llm_latency']:.0f}ms + TTS {metrics['tts_latency']:.0f}ms = {total:.0f}ms\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)

def shutdown(sig, frame):
    """Handle shutdown signal"""
    global running
    print("\n\n🛑 Shutting down...")
    running = False

# ==============================================================================
# --- ENTRY POINT ---
# ==============================================================================

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    if not initialize():
        print("\n❌ Initialization failed. Check errors above.")
        sys.exit(1)
    
    main_loop()
    
    print("\n🪐 Goodbye!\n")
