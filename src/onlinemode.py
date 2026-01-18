#!/usr/bin/env python3
"""
🪐 PLUTO - ONLINE MODE (Ultimate API Power)
Uses: Groq Whisper (STT) + Groq LLM + ElevenLabs (TTS)
Optimized for Raspberry Pi 4

Audio Config: Card 3 (USB PnP Sound Device / PCM2902)
- Microphone: hw:3,0 (mono, 48000Hz)
- Speaker: plughw:3,0 (stereo, auto-convert)
"""

import os
import sys
import io
import time
import wave
import tempfile
import subprocess
from collections import deque

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# ElevenLabs Voice Settings
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
ELEVENLABS_MODEL = "eleven_flash_v2_5"  # Fastest + cheapest model

# Groq Model Settings
GROQ_LLM_MODEL = "openai/gpt-oss-120b"      # GPT-OSS 120B
GROQ_STT_MODEL = "whisper-large-v3-turbo"   # Fast Whisper

# ==============================================================================
# --- AUDIO SETTINGS (Card 3 - USB PnP Sound Device) ---
# ==============================================================================

AUDIO_CARD = 3
AUDIO_DEVICE_MIC = f"hw:{AUDIO_CARD},0"      # Microphone (mono)
AUDIO_DEVICE_SPEAKER = f"plughw:{AUDIO_CARD},0"  # Speaker (auto-converts to stereo)

# Recording settings - optimized for speed
MIC_SAMPLE_RATE = 16000  # Lower sample rate = smaller file = faster upload
MIC_CHANNELS = 1
MIC_FORMAT = "S16_LE"

# ==============================================================================
# --- PLUTO'S PERSONALITY ---
# ==============================================================================

PLUTO_SYSTEM_PROMPT = """You are "Pluto," an AI-powered welcoming robot designed for organizations. You were 
created by Abdelrahman Mohamed (Nero) and Hamza Amgad, under the supervision of 
Dr. Amgad Byoumy.

Rules:
- Answer ONLY what the user asked - nothing more
- Keep responses short (1-2 sentences)
- Be practical, realistic, and professional
- Don't add random information or go off-topic
- If you don't know something, say "I'm not sure about that"
- Stay focused on being a helpful greeting assistant

You are warm and welcoming, but always stay on topic."""

# ==============================================================================
# --- IMPORTS ---
# ==============================================================================

def check_dependencies():
    """Check and report missing dependencies"""
    missing = []
    
    try:
        from groq import Groq
    except ImportError:
        missing.append("groq")
    
    try:
        from elevenlabs import ElevenLabs
    except ImportError:
        missing.append("elevenlabs")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        sys.exit(1)
    
    return True

check_dependencies()

from groq import Groq
from elevenlabs import ElevenLabs
import numpy as np

# ==============================================================================
# --- GLOBAL STATE ---
# ==============================================================================

groq_client = None
elevenlabs_client = None

# Conversation memory (last 5 turns)
conversation_history = deque(maxlen=10)

# Performance tracking
metrics = {
    "stt_latency": 0,
    "llm_latency": 0,
    "tts_latency": 0,
}

# ==============================================================================
# --- INITIALIZATION ---
# ==============================================================================

def initialize():
    """Initialize all API clients"""
    global groq_client, elevenlabs_client
    
    print("\n" + "="*60)
    print("🪐 PLUTO - ONLINE MODE")
    print("="*60 + "\n")
    
    print(f"Audio Configuration:")
    print(f"   Microphone: {AUDIO_DEVICE_MIC} (mono, {MIC_SAMPLE_RATE}Hz)")
    print(f"   Speaker:    {AUDIO_DEVICE_SPEAKER} (stereo)")
    print("")
    
    # Check API keys
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set!")
        print("   Run: export GROQ_API_KEY='your-key-here'")
        return False
    
    if not ELEVENLABS_API_KEY:
        print("❌ ELEVENLABS_API_KEY not set!")
        print("   Run: export ELEVENLABS_API_KEY='your-key-here'")
        return False
    
    try:
        # Initialize Groq
        print("🧠 Connecting to Groq...")
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Test Groq connection
        test = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model=GROQ_LLM_MODEL,
            max_completion_tokens=10,
            reasoning_effort="low"
        )
        print(f"✅ Groq connected ({GROQ_LLM_MODEL})")
        
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return False
    
    try:
        # Initialize ElevenLabs
        print("🔊 Connecting to ElevenLabs...")
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        print(f"✅ ElevenLabs connected ({ELEVENLABS_MODEL})")
        
    except Exception as e:
        print(f"❌ ElevenLabs error: {e}")
        return False
    
    print("\n✅ Ready!\n")
    return True

# ==============================================================================
# --- AUDIO RECORDING (using arecord) ---
# ==============================================================================

def record_audio_vad(max_duration=7, silence_duration=0.6):
    """Record audio with Voice Activity Detection"""
    
    print("🎤 Listening... (speak now)")
    
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = temp_wav.name
    temp_wav.close()
    
    # Record raw audio using arecord in background
    cmd = [
        'arecord',
        '-D', AUDIO_DEVICE_MIC,
        '-f', MIC_FORMAT,
        '-c', str(MIC_CHANNELS),
        '-r', str(MIC_SAMPLE_RATE),
        '-t', 'wav',
        temp_path
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Monitor for voice activity by checking file size growth
        speech_started = False
        silence_count = 0
        start_time = time.time()
        last_size = 0
        
        while time.time() - start_time < max_duration:
            time.sleep(0.1)  # Check more frequently
            
            if os.path.exists(temp_path):
                current_size = os.path.getsize(temp_path)
                growth_rate = current_size - last_size
                last_size = current_size
                
                # Simple VAD based on data growth (audio coming in)
                if growth_rate > 2000:  # Lower threshold for faster detection
                    if not speech_started:
                        print("🎙️  Speech detected...")
                        speech_started = True
                    silence_count = 0
                elif speech_started:
                    silence_count += 1
                    
                    # End after silence detected
                    if silence_count >= int(silence_duration / 0.1):
                        print("   End of speech")
                        break
        
        # Stop recording
        process.terminate()
        process.wait(timeout=2)
        
        # Check if we got audio
        if speech_started and os.path.exists(temp_path) and os.path.getsize(temp_path) > 5000:
            with open(temp_path, 'rb') as f:
                wav_data = f.read()
            
            os.unlink(temp_path)
            
            wav_buffer = io.BytesIO(wav_data)
            wav_buffer.seek(0)
            return wav_buffer
        else:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            print("   No speech detected")
            return None
            
    except Exception as e:
        print(f"❌ Recording error: {e}")
        process.terminate()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return None

# ==============================================================================
# --- STT: Speech-to-Text (Groq Whisper) ---
# ==============================================================================

def transcribe_audio(audio_buffer):
    """Transcribe audio using Groq Whisper API"""
    global metrics
    
    if audio_buffer is None:
        return None
    
    print("📝 Transcribing...")
    
    start_time = time.time()
    
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_buffer.read())
            wav_path = f.name
        
        # Compress to FLAC for faster upload (smaller file)
        flac_path = wav_path.replace('.wav', '.flac')
        compress_result = subprocess.run(
            ['ffmpeg', '-y', '-i', wav_path, '-ac', '1', '-ar', '16000', flac_path],
            capture_output=True, timeout=10
        )
        
        # Use compressed file if available, otherwise original
        if compress_result.returncode == 0 and os.path.exists(flac_path):
            upload_path = flac_path
        else:
            upload_path = wav_path
        
        # Transcribe with Groq Whisper
        with open(upload_path, 'rb') as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model=GROQ_STT_MODEL,
                language="en",
                response_format="text"
            )
        
        # Cleanup
        try:
            os.unlink(wav_path)
            if os.path.exists(flac_path):
                os.unlink(flac_path)
        except:
            pass
        
        metrics["stt_latency"] = (time.time() - start_time) * 1000
        
        text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
        
        if text:
            print(f"📝 \"{text}\" ({metrics['stt_latency']:.0f}ms)")
        
        return text if text else None
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None

# ==============================================================================
# --- LLM: Language Model (Groq) ---
# ==============================================================================

def get_response(user_text):
    """Get response from Groq LLM with conversation memory"""
    global metrics, conversation_history
    
    print("🧠 Thinking...")
    
    start_time = time.time()
    
    # Build messages with history
    messages = [{"role": "system", "content": PLUTO_SYSTEM_PROMPT}]
    
    # Add conversation history
    for msg in conversation_history:
        messages.append(msg)
    
    # Add current user message
    messages.append({"role": "user", "content": user_text})
    
    try:
        # Use streaming for faster perceived response
        response_text = ""
        
        stream = groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_LLM_MODEL,
            max_completion_tokens=256,
            temperature=0.3,  # Low = focused, no random tangents
            top_p=0.8,
            reasoning_effort="low",
            stream=True,
            stop=None
        )
        
        # Collect streamed response
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        
        metrics["llm_latency"] = (time.time() - start_time) * 1000
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": response_text})
        
        print(f"💬 \"{response_text}\" ({metrics['llm_latency']:.0f}ms)")
        
        return response_text
        
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return "I'm having trouble thinking right now. Could you try again?"

# ==============================================================================
# --- TTS: Text-to-Speech (ElevenLabs) ---
# ==============================================================================

def speak(text):
    """Convert text to speech using ElevenLabs with streaming playback"""
    global metrics
    
    print("🔊 Speaking...")
    
    start_time = time.time()
    
    try:
        # Start aplay process - it will play audio as we pipe it in
        wav_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        # Generate audio from ElevenLabs - stream directly
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id=ELEVENLABS_MODEL,
            output_format="pcm_24000"
        )
        
        # Collect chunks and write WAV file
        audio_chunks = []
        first_chunk_time = None
        
        for chunk in audio_generator:
            if first_chunk_time is None:
                first_chunk_time = time.time()
                metrics["tts_latency"] = (first_chunk_time - start_time) * 1000
                print(f"   First audio chunk: {metrics['tts_latency']:.0f}ms")
            audio_chunks.append(chunk)
        
        audio_data = b''.join(audio_chunks)
        
        # Write WAV file
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_data)
        
        # Play audio
        subprocess.run(
            ['aplay', '-D', AUDIO_DEVICE_SPEAKER, wav_path],
            capture_output=True, timeout=60
        )
        
        # Cleanup
        try:
            os.unlink(wav_path)
        except:
            pass
        
        total_time = (time.time() - start_time) * 1000
        print(f"   ✅ Total speak: {total_time:.0f}ms")
        
    except Exception as e:
        print(f"❌ TTS error: {e}")

# ==============================================================================
# --- MAIN LOOP ---
# ==============================================================================

def print_metrics():
    """Print performance summary"""
    total = metrics["stt_latency"] + metrics["llm_latency"] + metrics["tts_latency"]
    print(f"\n📊 Latency: STT {metrics['stt_latency']:.0f}ms + LLM {metrics['llm_latency']:.0f}ms + TTS {metrics['tts_latency']:.0f}ms = {total:.0f}ms\n")

def main():
    """Main conversation loop"""
    
    if not initialize():
        print("\n❌ Initialization failed. Check errors above.")
        sys.exit(1)
    
    # Greeting
    print("="*60)
    print("🚀 Pluto is ready!")
    print("="*60)
    print("\nPress Ctrl+C to stop\n")
    
    speak("Hello! I'm Pluto, your welcoming assistant. How can I help you today?")
    
    try:
        while True:
            # 1. Listen with VAD
            audio_buffer = record_audio_vad()
            
            if audio_buffer is None:
                continue
            
            # 2. Transcribe (Groq Whisper)
            user_text = transcribe_audio(audio_buffer)
            
            if not user_text:
                continue
            
            # 3. Get response (Groq LLM)
            response = get_response(user_text)
            
            # 4. Speak (ElevenLabs)
            speak(response)
            
            # 5. Show metrics
            print_metrics()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        speak("Goodbye! It was nice talking to you!")
        
    finally:
        print("\n🪐 Pluto offline. Goodbye!\n")

# ==============================================================================
# --- ENTRY POINT ---
# ==============================================================================

if __name__ == "__main__":
    main()
