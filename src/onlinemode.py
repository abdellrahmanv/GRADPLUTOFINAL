#!/usr/bin/env python3
"""
🪐 PLUTO - ONLINE MODE (Ultimate API Power)
Uses: Groq Whisper (STT) + Groq LLM + ElevenLabs (TTS)
Optimized for Raspberry Pi 4
"""

import os
import sys
import io
import time
import wave
import tempfile
import threading
from collections import deque

# ==============================================================================
# --- CONFIGURATION (Use Environment Variables!) ---
# ==============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# ElevenLabs Voice Settings
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
ELEVENLABS_MODEL = "eleven_turbo_v2_5"  # Fastest model

# Groq Model Settings
GROQ_LLM_MODEL = "llama-3.1-70b-versatile"  # Smart + Fast
GROQ_STT_MODEL = "whisper-large-v3-turbo"   # Fastest Whisper

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4096

# ==============================================================================
# --- PLUTO'S PERSONALITY ---
# ==============================================================================

PLUTO_SYSTEM_PROMPT = """You are "Pluto," an AI-powered welcoming robot designed for organizations. You were 
created by Abdelrahman Mohamed (Nero) and Hamza Amgad, under the supervision of 
Dr. Amgad Byoumy.

Your primary functions:
- Greet visitors warmly and make them comfortable
- Provide information about the organization
- Direct visitors to appropriate departments
- Answer FAQs professionally

Your personality:
- Warm, welcoming, professional
- Enthusiastic but not over-the-top
- Patient and understanding
- Clear and concise

Response guidelines:
- Keep responses brief (1-2 sentences for greetings, 2-3 for questions)
- Acknowledge every visitor
- For complex questions: "That's a great question! I'd recommend speaking with [department]. I can help you find them!"
- Stay in character as a friendly greeting robot

Always be cheerful and make visitors feel valued!"""

# ==============================================================================
# --- IMPORTS (with error handling) ---
# ==============================================================================

def check_dependencies():
    """Check and report missing dependencies"""
    missing = []
    
    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio")
    
    try:
        from groq import Groq
    except ImportError:
        missing.append("groq")
    
    try:
        from elevenlabs import ElevenLabs
    except ImportError:
        missing.append("elevenlabs")
    
    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        sys.exit(1)
    
    return True

check_dependencies()

import pyaudio
from groq import Groq
from elevenlabs import ElevenLabs

# ==============================================================================
# --- GLOBAL STATE ---
# ==============================================================================

groq_client = None
elevenlabs_client = None
audio = None

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
    global groq_client, elevenlabs_client, audio
    
    print("\n" + "="*60)
    print("🪐 PLUTO - ONLINE MODE")
    print("="*60 + "\n")
    
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
            max_tokens=5
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
    
    try:
        # Initialize PyAudio
        print("🎤 Initializing microphone...")
        audio = pyaudio.PyAudio()
        print("✅ Microphone ready")
        
    except Exception as e:
        print(f"❌ Audio error: {e}")
        return False
    
    print("\n✅ All systems online!\n")
    return True

# ==============================================================================
# --- STT: Speech-to-Text (Groq Whisper) ---
# ==============================================================================

def record_audio(max_duration=15, silence_threshold=500, silence_duration=1.5):
    """Record audio from microphone with voice activity detection"""
    
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("🎤 Listening...")
    
    frames = []
    is_speaking = False
    silence_chunks = 0
    silence_threshold_chunks = int(silence_duration * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks = int(max_duration * SAMPLE_RATE / CHUNK_SIZE)
    
    import numpy as np
    
    for _ in range(max_chunks):
        try:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            
            # Calculate audio energy
            audio_data = np.frombuffer(data, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            
            if energy > silence_threshold:
                if not is_speaking:
                    print("🎙️  Speech detected...")
                    is_speaking = True
                silence_chunks = 0
            elif is_speaking:
                silence_chunks += 1
                if silence_chunks >= silence_threshold_chunks:
                    print("   End of speech")
                    break
                    
        except Exception as e:
            print(f"⚠️  Audio error: {e}")
            break
    
    stream.stop_stream()
    stream.close()
    
    if not is_speaking:
        return None
    
    # Convert to WAV bytes
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
    
    wav_buffer.seek(0)
    return wav_buffer

def transcribe_audio(audio_buffer):
    """Transcribe audio using Groq Whisper API"""
    global metrics
    
    if audio_buffer is None:
        return None
    
    print("📝 Transcribing with Groq Whisper...")
    
    start_time = time.time()
    
    try:
        # Create temp file for Groq API
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_buffer.read())
            temp_path = f.name
        
        # Transcribe with Groq Whisper
        with open(temp_path, 'rb') as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model=GROQ_STT_MODEL,
                language="en",
                response_format="text"
            )
        
        # Cleanup
        os.unlink(temp_path)
        
        metrics["stt_latency"] = (time.time() - start_time) * 1000
        
        text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
        
        if text:
            print(f"📝 \"{text}\" ({metrics['stt_latency']:.0f}ms)")
        
        return text if text else None
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None

# ==============================================================================
# --- LLM: Language Model (Groq) with Streaming ---
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
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stream=True
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
# --- TTS: Text-to-Speech (ElevenLabs) with Streaming ---
# ==============================================================================

def speak(text):
    """Convert text to speech using ElevenLabs and play it"""
    global metrics
    
    print("🔊 Speaking...")
    
    start_time = time.time()
    
    try:
        # Generate audio with streaming
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id=ELEVENLABS_MODEL,
            output_format="pcm_22050"  # Raw PCM for faster playback
        )
        
        # Collect audio chunks
        audio_chunks = b''.join(audio_generator)
        
        metrics["tts_latency"] = (time.time() - start_time) * 1000
        
        # Play audio using PyAudio
        play_stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050,
            output=True
        )
        
        play_stream.write(audio_chunks)
        play_stream.stop_stream()
        play_stream.close()
        
        total_time = (time.time() - start_time) * 1000
        print(f"   TTS: {metrics['tts_latency']:.0f}ms, Total: {total_time:.0f}ms")
        
    except Exception as e:
        print(f"❌ TTS error: {e}")
        print(f"   Details: {type(e).__name__}: {str(e)}")

# ==============================================================================
# --- MAIN LOOP ---
# ==============================================================================

def print_metrics():
    """Print performance summary"""
    total = metrics["stt_latency"] + metrics["llm_latency"] + metrics["tts_latency"]
    print(f"\n📊 Latency: STT {metrics['stt_latency']:.0f}ms + LLM {metrics['llm_latency']:.0f}ms + TTS {metrics['tts_latency']:.0f}ms = {total:.0f}ms total\n")

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
            # 1. Listen
            audio_buffer = record_audio()
            
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
        if audio:
            audio.terminate()
        print("\n🪐 Pluto offline. Goodbye!\n")

# ==============================================================================
# --- ENTRY POINT ---
# ==============================================================================

if __name__ == "__main__":
    main()
