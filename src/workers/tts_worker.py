"""
🪐 PLUTO v2 - TTS Worker
Text-to-Speech using Piper (optimized for Raspberry Pi 4)
"""

import queue
import threading
import time
import subprocess
import tempfile
import wave
import os
from pathlib import Path

try:
    import pyaudio
except ImportError:
    print("❌ PyAudio not installed. Run: pip install pyaudio")
    raise

from config import PIPER_CONFIG, CACHE_DIR


class TTSWorker:
    """Text-to-Speech worker using Piper"""
    
    def __init__(self, input_queue: queue.Queue):
        self.input_queue = input_queue
        self.running = False
        self.thread = None
        self.audio = None
        
        # TTS cache for common phrases
        self.cache_dir = CACHE_DIR / "tts"
        self.cache: Dict[str, str] = {}  # text -> wav_path
        
    def initialize(self) -> bool:
        """Initialize Piper TTS"""
        try:
            # Check Piper binary
            piper_path = Path(PIPER_CONFIG['piper_binary'])
            if not piper_path.exists():
                print(f"   Piper not found at: {piper_path}")
                return False
            
            # Check model
            model_path = Path(PIPER_CONFIG['model_path'])
            if not model_path.exists():
                print(f"   Piper model not found at: {model_path}")
                return False
            
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            print(f"   Piper binary: {piper_path}")
            print(f"   Voice model: {model_path.name}")
            
            return True
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def warmup(self):
        """Warmup with test synthesis"""
        print("   TTS warmup...")
        try:
            start = time.time()
            self._synthesize("Hello")
            elapsed = (time.time() - start) * 1000
            print(f"   TTS warmup: {elapsed:.0f}ms")
        except Exception as e:
            print(f"   TTS warmup failed: {e}")
    
    def start(self):
        """Start TTS processing thread"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop TTS processing"""
        self.running = False
        
        if self.audio:
            self.audio.terminate()
        
        if self.thread:
            self.thread.join(timeout=2)
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get response from LLM
                item = self.input_queue.get(timeout=1.0)
                
                if item['type'] == 'response':
                    text = item['text']
                    
                    print(f"🔊 Speaking...")
                    
                    start = time.time()
                    wav_path = self._synthesize(text)
                    synth_time = (time.time() - start) * 1000
                    
                    if wav_path:
                        self._play_audio(wav_path)
                        
                        # Clean up temp file
                        if not str(wav_path).startswith(str(self.cache_dir)):
                            try:
                                os.unlink(wav_path)
                            except:
                                pass
                    
                    total_time = (time.time() - start) * 1000
                    print(f"   TTS: {synth_time:.0f}ms synth, {total_time:.0f}ms total")
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ TTS error: {e}")
    
    def _synthesize(self, text: str) -> str:
        """Synthesize text to WAV file"""
        
        # Check cache
        cache_key = text.lower().strip()[:50]
        if cache_key in self.cache:
            cached_path = self.cache[cache_key]
            if os.path.exists(cached_path):
                return cached_path
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
        
        try:
            # Run Piper
            cmd = [
                PIPER_CONFIG['piper_binary'],
                '--model', PIPER_CONFIG['model_path'],
                '--output_file', wav_path,
            ]
            
            process = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=30,
            )
            
            if process.returncode != 0:
                print(f"   Piper error: {process.stderr}")
                return None
            
            return wav_path
            
        except subprocess.TimeoutExpired:
            print("   Piper timeout")
            return None
        except Exception as e:
            print(f"   Synthesis error: {e}")
            return None
    
    def _play_audio(self, wav_path: str):
        """Play WAV file through speakers"""
        try:
            with wave.open(wav_path, 'rb') as wf:
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                )
                
                # Read and play in chunks
                chunk_size = 4096
                data = wf.readframes(chunk_size)
                
                while data:
                    stream.write(data)
                    data = wf.readframes(chunk_size)
                
                stream.stop_stream()
                stream.close()
                
        except Exception as e:
            print(f"   Playback error: {e}")
    
    def cache_phrase(self, text: str):
        """Pre-cache a common phrase"""
        cache_key = text.lower().strip()[:50]
        cache_path = self.cache_dir / f"{hash(cache_key)}.wav"
        
        if not cache_path.exists():
            wav_path = self._synthesize(text)
            if wav_path:
                os.rename(wav_path, cache_path)
        
        self.cache[cache_key] = str(cache_path)
