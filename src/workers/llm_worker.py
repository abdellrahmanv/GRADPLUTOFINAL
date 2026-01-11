"""
🪐 PLUTO v2 - LLM Worker
Language Model using Ollama + Qwen2.5 (optimized for Raspberry Pi 4)
"""

import queue
import threading
import time
import requests
from typing import List, Dict

from config import OLLAMA_CONFIG


class LLMWorker:
    """Language Model worker using Ollama"""
    
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False
        self.thread = None
        
        self.conversation_history: List[Dict[str, str]] = []
        self.api_url = f"{OLLAMA_CONFIG['host']}/api/generate"
        
    def initialize(self) -> bool:
        """Check Ollama server and model"""
        try:
            print(f"   Checking Ollama at {OLLAMA_CONFIG['host']}...")
            
            # Check server
            response = requests.get(
                f"{OLLAMA_CONFIG['host']}/api/tags",
                timeout=5
            )
            
            if response.status_code != 200:
                print("   Ollama server not responding")
                return False
            
            # Check model
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if OLLAMA_CONFIG['model'] not in model_names:
                # Try without tag
                base_model = OLLAMA_CONFIG['model'].split(':')[0]
                found = any(base_model in name for name in model_names)
                
                if not found:
                    print(f"   Model '{OLLAMA_CONFIG['model']}' not found")
                    print(f"   Available: {model_names}")
                    print(f"   Run: ollama pull {OLLAMA_CONFIG['model']}")
                    return False
            
            print(f"   Model: {OLLAMA_CONFIG['model']}")
            return True
            
        except requests.exceptions.ConnectionError:
            print(f"   Cannot connect to Ollama")
            print(f"   Run: ollama serve")
            return False
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    def warmup(self):
        """Warmup model with test inference"""
        print("   LLM warmup...")
        try:
            start = time.time()
            response = self._generate("Hello", max_tokens=10)
            elapsed = (time.time() - start) * 1000
            print(f"   LLM warmup: {elapsed:.0f}ms")
        except Exception as e:
            print(f"   LLM warmup failed: {e}")
    
    def start(self):
        """Start processing thread"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop processing"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get transcription from STT
                item = self.input_queue.get(timeout=1.0)
                
                if item['type'] == 'transcription':
                    user_text = item['text']
                    
                    print(f"🧠 Thinking...")
                    
                    start = time.time()
                    response_text = self._generate(user_text)
                    latency = (time.time() - start) * 1000
                    
                    print(f"💬 \"{response_text}\" ({latency:.0f}ms)")
                    
                    # Update history
                    self.conversation_history.append({'role': 'user', 'content': user_text})
                    self.conversation_history.append({'role': 'assistant', 'content': response_text})
                    
                    # Trim history
                    max_history = OLLAMA_CONFIG['max_history'] * 2
                    if len(self.conversation_history) > max_history:
                        self.conversation_history = self.conversation_history[-max_history:]
                    
                    # Send to TTS
                    self.output_queue.put({
                        'type': 'response',
                        'text': response_text,
                        'timestamp': time.time(),
                        'latency_ms': latency,
                    })
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ LLM error: {e}")
    
    def _generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response from Ollama"""
        
        # Build context from history
        context = ""
        for msg in self.conversation_history[-4:]:  # Last 2 turns
            role = "User" if msg['role'] == 'user' else "Assistant"
            context += f"{role}: {msg['content']}\n"
        
        full_prompt = f"{context}User: {prompt}\nAssistant:"
        
        # Trim if too long
        if len(full_prompt) > 500:
            full_prompt = f"User: {prompt}\nAssistant:"
        
        try:
            payload = {
                'model': OLLAMA_CONFIG['model'],
                'prompt': full_prompt,
                'system': OLLAMA_CONFIG['system_prompt'],
                'stream': False,
                'options': {
                    'temperature': OLLAMA_CONFIG['temperature'],
                    'top_p': OLLAMA_CONFIG['top_p'],
                    'num_predict': max_tokens or OLLAMA_CONFIG['max_tokens'],
                    'top_k': 40,
                    'repeat_penalty': 1.1,
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=OLLAMA_CONFIG['timeout']
            )
            response.raise_for_status()
            
            result = response.json()
            text = result.get('response', '').strip()
            
            # Limit to 2 sentences
            sentences = text.replace('!', '.').replace('?', '.').split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) > 2:
                text = '. '.join(sentences[:2]) + '.'
            
            return text
            
        except requests.exceptions.Timeout:
            return "I'm thinking too slowly. Please try again."
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama error: {e}")
            return "I encountered an error."
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return "Something went wrong."
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
