"""
🪐 PLUTO v2 - STT Worker
Speech-to-Text using faster-whisper (optimized for Raspberry Pi 4)
"""

import queue
import threading
import time
import numpy as np

try:
    import pyaudio
except ImportError:
    print("❌ PyAudio not installed. Run: pip install pyaudio")
    raise

from config import AUDIO_CONFIG, WHISPER_CONFIG


class STTWorker:
    """Speech-to-Text worker using faster-whisper"""
    
    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue
        self.running = False
        self.paused = False
        self.thread = None
        self.model = None
        self.audio = None
        self.stream = None
        
    def initialize(self) -> bool:
        """Initialize faster-whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            print(f"   Loading Whisper {WHISPER_CONFIG['model_size']} model...")
            
            self.model = WhisperModel(
                WHISPER_CONFIG['model_size'],
                device=WHISPER_CONFIG['device'],
                compute_type=WHISPER_CONFIG['compute_type'],
            )
            
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            return True
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def warmup(self):
        """Warmup model with silent audio"""
        print("   STT warmup...")
        try:
            # Create 1 second of silence
            silence = np.zeros(16000, dtype=np.float32)
            
            start = time.time()
            segments, _ = self.model.transcribe(
                silence,
                language=WHISPER_CONFIG['language'],
                beam_size=WHISPER_CONFIG['beam_size'],
                vad_filter=False,
            )
            list(segments)  # Force evaluation
            
            elapsed = (time.time() - start) * 1000
            print(f"   STT warmup: {elapsed:.0f}ms")
            
        except Exception as e:
            print(f"   STT warmup failed: {e}")
    
    def start(self):
        """Start listening thread"""
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop listening"""
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        if self.thread:
            self.thread.join(timeout=2)
    
    def pause(self):
        """Pause listening (while TTS is playing)"""
        self.paused = True
    
    def resume(self):
        """Resume listening"""
        self.paused = False
    
    def _listen_loop(self):
        """Main listening loop"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=AUDIO_CONFIG['channels'],
                rate=AUDIO_CONFIG['sample_rate'],
                input=True,
                frames_per_buffer=AUDIO_CONFIG['chunk_size'],
                input_device_index=AUDIO_CONFIG['input_device_index'],
            )
            
            print("🎤 Microphone active")
            
            audio_buffer = []
            is_speaking = False
            silence_count = 0
            
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                try:
                    # Read audio chunk
                    data = self.stream.read(
                        AUDIO_CONFIG['chunk_size'],
                        exception_on_overflow=False
                    )
                    
                    # Convert to numpy
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    
                    # Calculate energy
                    energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
                    
                    if energy > AUDIO_CONFIG['energy_threshold']:
                        # Speech detected
                        if not is_speaking:
                            is_speaking = True
                            audio_buffer = []
                            print("🎙️  Speech detected...")
                        
                        audio_buffer.append(audio_chunk)
                        silence_count = 0
                        
                    elif is_speaking:
                        # Silence during speech
                        audio_buffer.append(audio_chunk)
                        silence_count += 1
                        
                        # Check if speech ended
                        silence_threshold = int(
                            AUDIO_CONFIG['silence_duration'] * 
                            AUDIO_CONFIG['sample_rate'] / 
                            AUDIO_CONFIG['chunk_size']
                        )
                        
                        if silence_count >= silence_threshold:
                            # Speech ended, transcribe
                            is_speaking = False
                            
                            if len(audio_buffer) > 0:
                                self._transcribe(audio_buffer)
                            
                            audio_buffer = []
                            silence_count = 0
                
                except Exception as e:
                    print(f"⚠️  Audio read error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Listen loop error: {e}")
    
    def _transcribe(self, audio_buffer):
        """Transcribe audio buffer"""
        try:
            # Combine chunks
            audio_data = np.concatenate(audio_buffer)
            
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Check minimum duration
            duration = len(audio_float) / AUDIO_CONFIG['sample_rate']
            if duration < AUDIO_CONFIG['min_phrase_duration']:
                return
            
            print(f"   Transcribing {duration:.1f}s of audio...")
            
            start = time.time()
            
            segments, info = self.model.transcribe(
                audio_float,
                language=WHISPER_CONFIG['language'],
                beam_size=WHISPER_CONFIG['beam_size'],
                vad_filter=WHISPER_CONFIG['vad_filter'],
            )
            
            # Get text
            text = ""
            for segment in segments:
                text += segment.text
            
            text = text.strip()
            
            latency = (time.time() - start) * 1000
            
            if text:
                print(f"📝 \"{text}\" ({latency:.0f}ms)")
                
                # Send to LLM
                self.output_queue.put({
                    'type': 'transcription',
                    'text': text,
                    'timestamp': time.time(),
                    'latency_ms': latency,
                })
            else:
                print("   (no speech detected)")
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
