#!/usr/bin/env python3
"""
🪐 PLUTO v2 - Main Entry Point
Optimized Voice Assistant for Raspberry Pi 4
"""

import sys
import signal
import time
import queue
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import OLLAMA_CONFIG, WHISPER_CONFIG, PIPER_CONFIG
from workers.stt_worker import STTWorker
from workers.llm_worker import LLMWorker
from workers.tts_worker import TTSWorker


class PlutoAssistant:
    """Main voice assistant orchestrator"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("🪐 PLUTO v2 - Voice Assistant")
        print("="*60 + "\n")
        
        # Queues for worker communication
        self.stt_to_llm = queue.Queue(maxsize=5)
        self.llm_to_tts = queue.Queue(maxsize=5)
        
        # Workers
        self.stt_worker = None
        self.llm_worker = None
        self.tts_worker = None
        
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize all workers"""
        print("Initializing workers...\n")
        
        try:
            # STT Worker
            print("🎤 STT Worker (faster-whisper)...")
            self.stt_worker = STTWorker(self.stt_to_llm)
            if not self.stt_worker.initialize():
                print("❌ STT initialization failed")
                return False
            print("✅ STT ready\n")
            
            # LLM Worker
            print(f"🧠 LLM Worker ({OLLAMA_CONFIG['model']})...")
            self.llm_worker = LLMWorker(self.stt_to_llm, self.llm_to_tts)
            if not self.llm_worker.initialize():
                print("❌ LLM initialization failed")
                return False
            print("✅ LLM ready\n")
            
            # TTS Worker
            print("🔊 TTS Worker (Piper)...")
            self.tts_worker = TTSWorker(self.llm_to_tts)
            if not self.tts_worker.initialize():
                print("❌ TTS initialization failed")
                return False
            print("✅ TTS ready\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return False
    
    def warmup(self):
        """Warmup models for faster first response"""
        print("Warming up models...")
        
        if self.stt_worker:
            self.stt_worker.warmup()
        if self.llm_worker:
            self.llm_worker.warmup()
        if self.tts_worker:
            self.tts_worker.warmup()
            
        print("✅ Warmup complete\n")
    
    def start(self):
        """Start all workers"""
        print("="*60)
        print("🚀 Starting Pluto...")
        print("="*60 + "\n")
        
        self.running = True
        
        # Start workers
        self.stt_worker.start()
        self.llm_worker.start()
        self.tts_worker.start()
        
        print("✅ All workers running")
        print("\n🎤 Listening... (Press Ctrl+C to stop)\n")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    
    def stop(self):
        """Stop all workers"""
        print("\n\n🛑 Shutting down...")
        
        self.running = False
        
        if self.stt_worker:
            self.stt_worker.stop()
        if self.llm_worker:
            self.llm_worker.stop()
        if self.tts_worker:
            self.tts_worker.stop()
        
        print("✅ Shutdown complete")
        print("\n🪐 Goodbye!\n")


def main():
    """Main entry point"""
    assistant = PlutoAssistant()
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        assistant.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize
    if not assistant.initialize():
        print("\n❌ Failed to initialize. Check errors above.")
        sys.exit(1)
    
    # Warmup
    assistant.warmup()
    
    # Run
    assistant.start()


if __name__ == "__main__":
    main()
