#!/usr/bin/env python3
"""
🔊 PLUTO - Audio Diagnostic Tool
Run this to check your microphone and speakers on Raspberry Pi
"""

import subprocess
import sys

def run_cmd(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    except:
        return "Command failed"

print("\n" + "="*60)
print("🔊 PLUTO Audio Diagnostics")
print("="*60 + "\n")

# ============================================================================
# 1. Check ALSA devices
# ============================================================================
print("📋 ALSA Playback Devices (Speakers):")
print("-"*40)
print(run_cmd("aplay -l"))

print("\n📋 ALSA Recording Devices (Microphones):")
print("-"*40)
print(run_cmd("arecord -l"))

# ============================================================================
# 2. Check PulseAudio
# ============================================================================
print("\n📋 PulseAudio Sinks (Outputs):")
print("-"*40)
print(run_cmd("pactl list short sinks 2>/dev/null || echo 'PulseAudio not running'"))

print("\n📋 PulseAudio Sources (Inputs):")
print("-"*40)
print(run_cmd("pactl list short sources 2>/dev/null || echo 'PulseAudio not running'"))

# ============================================================================
# 3. Check PyAudio devices
# ============================================================================
print("\n📋 PyAudio Devices:")
print("-"*40)

try:
    import pyaudio
    p = pyaudio.PyAudio()
    
    print(f"Total devices: {p.get_device_count()}\n")
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        device_type = []
        if info['maxInputChannels'] > 0:
            device_type.append("🎤 INPUT")
        if info['maxOutputChannels'] > 0:
            device_type.append("🔊 OUTPUT")
        
        print(f"[{i}] {info['name']}")
        print(f"    Type: {' + '.join(device_type)}")
        print(f"    Sample Rate: {int(info['defaultSampleRate'])}Hz")
        print(f"    Input Channels: {info['maxInputChannels']}")
        print(f"    Output Channels: {info['maxOutputChannels']}")
        print()
    
    p.terminate()
    
except ImportError:
    print("PyAudio not installed. Run: pip install pyaudio")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# 4. Test speaker
# ============================================================================
print("\n" + "="*60)
print("🔊 SPEAKER TEST")
print("="*60)
print("\nDo you want to test your speakers? (y/n): ", end="")
try:
    if input().lower() == 'y':
        print("Playing test sound...")
        # Try different methods
        result = subprocess.run("speaker-test -t sine -f 440 -l 1 -p 1 2>/dev/null", 
                               shell=True, timeout=3)
        if result.returncode != 0:
            subprocess.run("aplay /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null", 
                          shell=True)
except KeyboardInterrupt:
    pass
except:
    print("Speaker test failed")

# ============================================================================
# 5. Test microphone
# ============================================================================
print("\n" + "="*60)
print("🎤 MICROPHONE TEST")
print("="*60)
print("\nDo you want to test your microphone? (y/n): ", end="")
try:
    if input().lower() == 'y':
        print("Recording 3 seconds... Speak now!")
        subprocess.run("arecord -d 3 -f cd /tmp/test_mic.wav 2>/dev/null", shell=True)
        print("Playing back...")
        subprocess.run("aplay /tmp/test_mic.wav 2>/dev/null", shell=True)
except KeyboardInterrupt:
    pass
except:
    print("Microphone test failed")

# ============================================================================
# 6. Recommendations
# ============================================================================
print("\n" + "="*60)
print("💡 RECOMMENDATIONS")
print("="*60)
print("""
If audio doesn't work, try:

1. Check volume:
   alsamixer

2. Set default audio device:
   sudo raspi-config
   → System Options → Audio → Choose output

3. For USB microphone/speaker:
   # List USB devices
   lsusb
   
   # Set USB audio as default
   echo 'pcm.!default {
       type asym
       playback.pcm "plughw:1,0"
       capture.pcm "plughw:1,0"
   }' > ~/.asoundrc

4. Restart audio:
   pulseaudio -k
   pulseaudio --start

5. Check connections and make sure device is plugged in!
""")
