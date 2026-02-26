# Timeline Testing — Reproduce All Version Latencies

This folder contains scripts to benchmark **every version** of the Pluto voice assistant
on the Raspberry Pi 4, using the **exact same engines and configurations** as each version.

## Quick Start (on the Pi)

```bash
# 1. Copy this folder to the Pi (or git pull)
cd ~/GRADPLUTOFINAL/timelinetesting

# 2. Install all models and dependencies
chmod +x setup.sh
./setup.sh          # uses existing system packages, downloads models

# 3. Run the benchmark (V3-opt + V4, 10 trials each)
python3 benchmark_timeline.py --skip-openai

# 4. Or run all versions (only if PyTorch works — unlikely on RPi4)
python3 benchmark_timeline.py

# 5. Or test specific versions
python3 benchmark_timeline.py --versions V3-opt V4
```

## What Gets Tested

| Version | STT Engine | STT Model | LLM | LLM Config |
|---------|-----------|-----------|-----|------------|
| V1 | OpenAI Whisper (PyTorch FP32) | base | None (keyword) | — |
| V2 | OpenAI Whisper (PyTorch FP32) | base | None (keyword) | — |
| V3 | OpenAI Whisper (PyTorch FP32) | tiny | Qwen2.5 0.5B q4_k_M | stream=False, 150 tokens |
| V3-opt | faster-whisper (CTranslate2 INT8) | tiny | Qwen2.5 0.5B q2_K | stream=False, 60 tokens |
| V4 | faster-whisper (CTranslate2 INT8) | base | Qwen2.5 0.5B q2_K | stream=True, 80 tokens |

## Output

Results are saved to `results/`:
- `timeline_YYYYMMDD_HHMMSS.json` — summary with mean ± std per version
- `timeline_full_YYYYMMDD_HHMMSS.json` — includes per-trial data
- `timeline_YYYYMMDD_HHMMSS.csv` — per-trial CSV for plotting

The script also prints a Markdown table and LaTeX table ready to paste into the paper.

## Notes

- V1 and V2 use the same STT engine; V2 only changed audio I/O (not benchmarked here since we use a WAV file, not live mic). Their STT/TTS numbers will be identical.
- OpenAI Whisper requires PyTorch (~800MB) which crashes on RPi4 due to ARM instruction incompatibility. Use `--skip-openai` on RPi4.
- The setup script reuses your existing system Python packages (faster-whisper, numpy, etc.) — no venv needed.
- Make sure `ollama serve` is running before testing.
- The CPU governor is set to `performance` by `setup.sh` for consistent results.
