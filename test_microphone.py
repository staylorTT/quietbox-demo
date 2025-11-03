#!/usr/bin/env python3
"""
Microphone Audio Quality Test Script

Tests microphone recording quality and plays it back to verify audio is working.
Helps diagnose issues with wake word detection.
"""

import sounddevice as sd
import numpy as np
import time
import wave
import sys

def test_microphone(device=None, duration=3, samplerate=16000):
    """Test microphone recording and playback"""
    
    print(f"Testing microphone audio quality...")
    print(f"Using device: {device if device is not None else 'default'}")
    print(f"Recording {duration} seconds...")
    print("Speak clearly into your microphone now!")
    print("(Recording will start in 1 second...)\n")
    time.sleep(1)
    
    try:
        # Record audio
        print("🔴 Recording...")
        recording = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype='float32',
            device=device
        )
        sd.wait()  # Wait until recording is finished
        print("✓ Recording complete\n")
        
        # Analyze audio quality
        audio_rms = np.sqrt(np.mean(recording**2))
        max_val = np.abs(recording).max()
        min_val = recording.min()
        
        print("=" * 60)
        print("Audio Quality Analysis:")
        print("=" * 60)
        print(f"  RMS level:     {audio_rms:.6f}")
        print(f"  Peak level:    {max_val:.6f}")
        print(f"  Min level:     {min_val:.6f}")
        print(f"  Sample rate:   {samplerate} Hz")
        print(f"  Duration:      {duration} seconds")
        print(f"  Samples:       {len(recording)}")
        print()
        
        # Quality assessment
        if audio_rms < 0.001:
            print("⚠️  WARNING: Audio level very low!")
            print("   - Microphone may not be working properly")
            print("   - Check microphone permissions")
            print("   - Check microphone is not muted")
            print("   - Try speaking louder or adjusting mic gain")
        elif audio_rms > 0.5:
            print("⚠️  WARNING: Audio level very high!")
            print("   - Audio may be clipping/distorted")
            print("   - Lower microphone input gain")
        else:
            print("✓ Audio levels look reasonable")
            if 0.01 <= audio_rms <= 0.3:
                print("  (Normal speech range detected)")
        
        # Play back the recording
        print("\n" + "=" * 60)
        print("Playing back your recording...")
        print("=" * 60)
        sd.play(recording, samplerate=samplerate)
        sd.wait()
        print("✓ Playback complete")
        print("\nCould you hear yourself clearly?")
        
        # Save to file
        filename = "test_recording.wav"
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(samplerate)
            # Convert float32 to int16
            recording_int16 = (recording * 32767).astype(np.int16)
            wf.writeframes(recording_int16.tobytes())
        
        print(f"\n✓ Saved to {filename}")
        print(f"  Play with: aplay {filename}")
        print(f"  Or: paplay {filename}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def list_devices():
    """List all available audio input devices"""
    print("=" * 60)
    print("Available Input Devices:")
    print("=" * 60)
    
    devices = sd.query_devices()
    default = sd.query_devices(kind='input')
    
    for dev in devices:
        if dev['max_input_channels'] > 0:
            marker = " <-- DEFAULT" if dev['index'] == default['index'] else ""
            print(f"  [{dev['index']}] {dev['name']}{marker}")
            print(f"      Channels: {dev['max_input_channels']}, "
                  f"Sample rate: {dev['default_samplerate']} Hz")
    
    print()
    print(f"Default input: [{default['index']}] {default['name']}")
    print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test microphone audio quality")
    parser.add_argument("--device", type=int, help="Device index to use (see --list)")
    parser.add_argument("--duration", type=int, default=3, help="Recording duration in seconds (default: 3)")
    parser.add_argument("--list", action="store_true", help="List all available input devices")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    
    args = parser.parse_args()
    
    if args.list:
        list_devices()
        sys.exit(0)
    
    print("Microphone Audio Quality Test")
    print("=" * 60)
    print()
    
    success = test_microphone(
        device=args.device,
        duration=args.duration,
        samplerate=args.samplerate
    )
    
    if success:
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        print("\nIf audio quality was good but wake words don't work,")
        print("the issue is likely with wake word recognition, not audio.")
    else:
        print("\n" + "=" * 60)
        print("Test failed - check audio device and permissions")
        print("=" * 60)
        sys.exit(1)

