#!/usr/bin/env python3
"""
Voice Assistant Benchmark — N=10 trials
Run on Raspberry Pi 4 to collect publishable latency metrics.

Measures per interaction:
  1. STT latency  (faster-whisper transcription)
  2. LLM latency  (Ollama first-token + full response)
  3. TTS latency  (Piper synthesis, excludes playback)
  4. Total pipeline latency

Two modes:
  --live     : Records real mic audio for each interaction (you speak 10 times)
  --synthetic: Uses a pre-recorded WAV file (repeatable, no mic needed)

Usage:
  python3 benchmark_voice.py --live --trials 10
  python3 benchmark_voice.py --synthetic test_audio.wav --trials 10
  python3 benchmark_voice.py --synthetic-text "What is the weather today" --trials 10
"""

import os
import sys
import time
import json
import wave
import argparse
import tempfile
import subprocess
import signal
import csv
from datetime import datetime

import numpy as np

# ============================================
# CONFIG — must match offlinemode.py exactly
# ============================================
AUDIO_CARD = 3
AUDIO_DEVICE_MIC = "hw:3,0"
AUDIO_DEVICE_SPEAKER = "plughw:3,0"
MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:0.5b-instruct-q2_k"
MAX_TOKENS = 80

PIPER_BINARY = os.path.expanduser("~/pluto-v2/piper/piper")
PIPER_MODEL = os.path.expanduser("~/pluto-v2/models/en_US-lessac-medium.onnx")

WHISPER_MODEL_SIZE = "base"
WHISPER_COMPUTE_TYPE = "int8"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================
# Components
# ============================================

whisper_model = None

def init_whisper():
    """Initialise faster-whisper model"""
    global whisper_model
    from faster_whisper import WhisperModel
    
    print(f"Loading faster-whisper ({WHISPER_MODEL_SIZE}, {WHISPER_COMPUTE_TYPE})...")
    start = time.time()
    
    cache_dir = os.path.expanduser("~/.cache/whisper")
    os.makedirs(cache_dir, exist_ok=True)
    
    whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root=cache_dir,
    )
    
    # Warmup
    silence = np.zeros(16000, dtype=np.float32)
    segments, _ = whisper_model.transcribe(silence, language="en", beam_size=1)
    list(segments)
    
    elapsed = (time.time() - start) * 1000
    print(f"  Whisper ready ({elapsed:.0f}ms warmup)")


def record_audio_fixed(duration_s=3):
    """Record audio from mic for fixed duration (matches offlinemode.py)"""
    temp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    cmd = [
        "arecord", "-D", AUDIO_DEVICE_MIC,
        "-f", "S16_LE", "-r", str(MIC_SAMPLE_RATE),
        "-c", str(MIC_CHANNELS), "-t", "wav",
        "-d", str(duration_s), "-q", temp_path,
    ]
    
    try:
        subprocess.run(cmd, timeout=duration_s + 5, capture_output=True)
    except Exception as e:
        print(f"  Recording error: {e}")
        return None
    
    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
        return temp_path
    return None


def transcribe_audio(audio_path):
    """Transcribe audio and return (text, latency_ms)"""
    start = time.perf_counter()
    
    with wave.open(audio_path, "rb") as wf:
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        frames = wf.readframes(n_frames)
    
    audio = np.frombuffer(frames, dtype=np.int16)
    if n_channels == 2:
        audio = audio[::2]
    audio_float = audio.astype(np.float32) / 32768.0
    
    # Resample if needed
    if sample_rate != 16000:
        duration_sec = len(audio_float) / sample_rate
        target_samples = int(duration_sec * 16000)
        indices = np.linspace(0, len(audio_float) - 1, target_samples)
        audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float).astype(np.float32)
    
    # Limit to 5s
    max_samples = 5 * 16000
    if len(audio_float) > max_samples:
        audio_float = audio_float[:max_samples]
    
    # Normalise
    max_val = np.max(np.abs(audio_float))
    if max_val > 0:
        audio_float = audio_float / max_val * 0.95
    
    segments, info = whisper_model.transcribe(
        audio_float,
        language="en",
        beam_size=1,
        best_of=1,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)
    text = "".join(text_parts).strip()
    
    latency_ms = (time.perf_counter() - start) * 1000
    return text, latency_ms


def query_llm(user_text):
    """Send query to Ollama, return (response_text, latency_ms, first_token_ms)"""
    import requests
    
    prompt = f"You are Pluto, a friendly robot assistant. Keep answers short.\n\nUser: {user_text}\nPluto:"
    
    start = time.perf_counter()
    first_token_time = None
    
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": MAX_TOKENS,
                "temperature": 0.3,
                "top_p": 0.8,
                "num_ctx": 512,
            },
        },
        timeout=30,
        stream=True,
    )
    
    result = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            result += token
            
            if first_token_time is None and token:
                first_token_time = time.perf_counter()
            
            if data.get("done", False):
                break
    
    total_ms = (time.perf_counter() - start) * 1000
    ft_ms = (first_token_time - start) * 1000 if first_token_time else total_ms
    
    result = result.strip().split("\n")[0].strip()
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    
    return result, total_ms, ft_ms


def synthesize_speech(text):
    """Run Piper TTS, return (wav_path, latency_ms)"""
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    start = time.perf_counter()
    result = subprocess.run(
        [PIPER_BINARY, "--model", PIPER_MODEL, "--output_file", wav_path],
        input=text, text=True, capture_output=True, timeout=30,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    
    if result.returncode != 0:
        print(f"  Piper error: {result.stderr}")
        return None, latency_ms
    
    return wav_path, latency_ms


def play_audio(wav_path):
    """Play WAV via aplay (blocking)"""
    try:
        subprocess.run(
            ["aplay", "-D", AUDIO_DEVICE_SPEAKER, wav_path],
            capture_output=True, timeout=60,
        )
    except:
        pass


def create_synthetic_wav(text_for_whisper="Hello, what is the time?"):
    """Create a synthetic WAV with Piper so Whisper has something to transcribe"""
    wav_path, _ = synthesize_speech(text_for_whisper)
    if wav_path and os.path.exists(wav_path):
        return wav_path
    return None


# ============================================
# Benchmark
# ============================================

def run_trial(trial_num, audio_path=None, play=False, synthetic_text=None):
    """
    Run one full interaction: Record/load → STT → LLM → TTS
    Returns metrics dict or None on failure.
    """
    print(f"\n--- Trial {trial_num} ---")
    metrics = {}
    
    # 1. Get audio
    if audio_path:
        # Use provided file
        temp_audio = audio_path
        own_audio = False
    elif synthetic_text:
        # Generate speech from text, then transcribe it back (round-trip test)
        print(f"  Generating synthetic audio for: \"{synthetic_text}\"")
        temp_audio = create_synthetic_wav(synthetic_text)
        own_audio = True
        if not temp_audio:
            print("  Failed to create synthetic audio")
            return None
    else:
        # Live recording
        print("  🎤 Speak now! (3 seconds)")
        temp_audio = record_audio_fixed(duration_s=3)
        own_audio = True
        if not temp_audio:
            print("  No audio captured")
            return None
    
    # 2. STT
    text, stt_ms = transcribe_audio(temp_audio)
    metrics["stt_ms"] = round(stt_ms, 1)
    print(f"  STT: \"{text}\" ({stt_ms:.0f}ms)")
    
    if own_audio and temp_audio != audio_path:
        try:
            os.unlink(temp_audio)
        except:
            pass
    
    if not text:
        print("  Empty transcription — skipping")
        return None
    
    # Use the synthetic_text as LLM input if transcription was garbled
    llm_input = text
    if synthetic_text and (not text or len(text) < 3):
        llm_input = synthetic_text
    
    # 3. LLM
    response, llm_ms, ft_ms = query_llm(llm_input)
    metrics["llm_ms"] = round(llm_ms, 1)
    metrics["llm_first_token_ms"] = round(ft_ms, 1)
    print(f"  LLM: \"{response}\" ({llm_ms:.0f}ms, first token {ft_ms:.0f}ms)")
    
    if not response:
        print("  No LLM response — skipping")
        return None
    
    # 4. TTS
    wav_path, tts_ms = synthesize_speech(response)
    metrics["tts_ms"] = round(tts_ms, 1)
    print(f"  TTS: {tts_ms:.0f}ms")
    
    # Total
    total = stt_ms + llm_ms + tts_ms
    metrics["total_ms"] = round(total, 1)
    metrics["user_text"] = llm_input
    metrics["llm_response"] = response
    
    print(f"  TOTAL: {total:.0f}ms  (STT {stt_ms:.0f} + LLM {llm_ms:.0f} + TTS {tts_ms:.0f})")
    
    # Play audio if requested
    if play and wav_path:
        play_audio(wav_path)
    
    # Cleanup TTS wav
    if wav_path:
        try:
            os.unlink(wav_path)
        except:
            pass
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Voice Assistant Benchmark")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials (default: 10)")
    parser.add_argument("--live", action="store_true", help="Record live from mic each trial")
    parser.add_argument("--synthetic", type=str, default=None, help="Path to pre-recorded WAV file")
    parser.add_argument("--synthetic-text", type=str, default=None,
                        help="Generate speech from this text, then benchmark the pipeline")
    parser.add_argument("--play", action="store_true", help="Play TTS output after each trial")
    args = parser.parse_args()
    
    # Default: use synthetic text if no mode specified
    if not args.live and not args.synthetic and not args.synthetic_text:
        args.synthetic_text = "Hello, what is the weather today?"
    
    # Predefined queries for variety in synthetic mode
    queries = [
        "Hello, what is the weather today?",
        "Tell me a fun fact about robots.",
        "What time is it?",
        "How does a computer work?",
        "What is your name?",
        "Tell me a short joke.",
        "What is artificial intelligence?",
        "How far is the moon?",
        "What is the capital of Egypt?",
        "Say something nice.",
    ]
    
    print("=" * 60)
    print("  Voice Assistant Benchmark")
    print(f"  STT:  faster-whisper {WHISPER_MODEL_SIZE} ({WHISPER_COMPUTE_TYPE})")
    print(f"  LLM:  Ollama {OLLAMA_MODEL}")
    print(f"  TTS:  Piper")
    print(f"  Mode: {'live mic' if args.live else 'synthetic'}")
    print(f"  Trials: {args.trials}")
    print("=" * 60)
    
    # Init
    init_whisper()
    
    # Verify Ollama
    import requests
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        assert r.status_code == 200
        print("  Ollama: connected")
    except:
        print("ERROR: Ollama not reachable. Start it with: ollama serve")
        sys.exit(1)
    
    # Verify Piper
    if not os.path.exists(PIPER_BINARY):
        print(f"ERROR: Piper not found at {PIPER_BINARY}")
        sys.exit(1)
    
    # Run trials
    trial_results = []
    for t in range(1, args.trials + 1):
        if args.live:
            result = run_trial(t, play=args.play)
        elif args.synthetic:
            result = run_trial(t, audio_path=args.synthetic, play=args.play)
        else:
            # Use variety of queries
            query = queries[(t - 1) % len(queries)]
            result = run_trial(t, synthetic_text=query, play=args.play)
        
        if result:
            result["trial"] = t
            trial_results.append(result)
        
        time.sleep(0.5)  # Brief pause between trials
    
    if not trial_results:
        print("\nNo successful trials.")
        sys.exit(1)
    
    # ---- Aggregate ----
    stt_vals = [r["stt_ms"] for r in trial_results]
    llm_vals = [r["llm_ms"] for r in trial_results]
    tts_vals = [r["tts_ms"] for r in trial_results]
    total_vals = [r["total_ms"] for r in trial_results]
    ft_vals = [r["llm_first_token_ms"] for r in trial_results]
    
    print("\n" + "=" * 60)
    print(f"  AGGREGATE RESULTS  (N={len(trial_results)} trials)")
    print("=" * 60)
    print(f"  STT:          {np.mean(stt_vals):7.1f} ± {np.std(stt_vals):.1f} ms")
    print(f"  LLM:          {np.mean(llm_vals):7.1f} ± {np.std(llm_vals):.1f} ms")
    print(f"  LLM 1st tok:  {np.mean(ft_vals):7.1f} ± {np.std(ft_vals):.1f} ms")
    print(f"  TTS:          {np.mean(tts_vals):7.1f} ± {np.std(tts_vals):.1f} ms")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL:        {np.mean(total_vals):7.1f} ± {np.std(total_vals):.1f} ms")
    print("=" * 60)
    
    # ---- Save results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "timestamp": timestamp,
        "config": {
            "stt_engine": "faster-whisper",
            "stt_model": WHISPER_MODEL_SIZE,
            "stt_compute": WHISPER_COMPUTE_TYPE,
            "llm_model": OLLAMA_MODEL,
            "llm_max_tokens": MAX_TOKENS,
            "tts_engine": "Piper",
            "trials": args.trials,
            "mode": "live" if args.live else "synthetic",
        },
        "aggregate": {
            "stt_mean_ms": round(float(np.mean(stt_vals)), 1),
            "stt_std_ms": round(float(np.std(stt_vals)), 1),
            "llm_mean_ms": round(float(np.mean(llm_vals)), 1),
            "llm_std_ms": round(float(np.std(llm_vals)), 1),
            "llm_first_token_mean_ms": round(float(np.mean(ft_vals)), 1),
            "llm_first_token_std_ms": round(float(np.std(ft_vals)), 1),
            "tts_mean_ms": round(float(np.mean(tts_vals)), 1),
            "tts_std_ms": round(float(np.std(tts_vals)), 1),
            "total_mean_ms": round(float(np.mean(total_vals)), 1),
            "total_std_ms": round(float(np.std(total_vals)), 1),
        },
        "trials": trial_results,
    }
    
    json_path = os.path.join(RESULTS_DIR, f"voice_benchmark_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")
    
    # CSV
    csv_path = os.path.join(RESULTS_DIR, f"voice_per_trial_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial", "stt_ms", "llm_ms", "llm_first_token_ms", "tts_ms", "total_ms",
            "user_text", "llm_response",
        ])
        writer.writeheader()
        for r in trial_results:
            writer.writerow(r)
    print(f"Per-trial CSV: {csv_path}")
    
    # LaTeX row
    print("\n--- Copy-paste for paper (LaTeX table row) ---")
    print(f"V4 (Final) & "
          f"{np.mean(stt_vals):.0f} ± {np.std(stt_vals):.0f} & "
          f"{np.mean(llm_vals):.0f} ± {np.std(llm_vals):.0f} & "
          f"{np.mean(tts_vals):.0f} ± {np.std(tts_vals):.0f} & "
          f"{np.mean(total_vals):.0f} ± {np.std(total_vals):.0f} \\\\")


if __name__ == "__main__":
    main()
