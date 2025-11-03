#!/usr/bin/env python3
"""
Test different beep sounds for the QuietBox ready chime
Allows you to try different frequencies, waveforms, and patterns to find a pleasant sound
"""
import numpy as np
import sounddevice as sd
import time

def generate_tone(frequency=800, duration=0.2, sample_rate=44100, waveform='sine', fade_in=0.01, fade_out=0.05, volume=0.3):
    """
    Generate a tone with various waveform options
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Audio sample rate
        waveform: 'sine', 'triangle', 'square', 'bell', 'chime'
        fade_in: Fade in time (seconds)
        fade_out: Fade out time (seconds)
        volume: Volume level (0.0 to 1.0)
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    if waveform == 'sine':
        wave = np.sin(2 * np.pi * frequency * t)
    elif waveform == 'triangle':
        wave = 2 * np.abs(2 * ((frequency * t) % 1) - 1) - 1
    elif waveform == 'square':
        wave = np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == 'bell':
        # Bell-like sound with harmonics
        wave = (np.sin(2 * np.pi * frequency * t) + 
                0.5 * np.sin(2 * np.pi * frequency * 2 * t) +
                0.25 * np.sin(2 * np.pi * frequency * 3 * t))
        wave = wave / 1.75  # Normalize
    elif waveform == 'chime':
        # Pleasant chime with multiple frequencies
        wave = (np.sin(2 * np.pi * frequency * t) +
                0.6 * np.sin(2 * np.pi * frequency * 1.5 * t) +
                0.3 * np.sin(2 * np.pi * frequency * 2 * t))
        wave = wave / 1.9  # Normalize
    else:
        wave = np.sin(2 * np.pi * frequency * t)
    
    # Apply fade in/out for smoother sound
    fade_in_samples = int(sample_rate * fade_in)
    fade_out_samples = int(sample_rate * fade_out)
    
    if fade_in_samples > 0:
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        wave[:fade_in_samples] *= fade_in_curve
    
    if fade_out_samples > 0:
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        wave[-fade_out_samples:] *= fade_out_curve
    
    # Apply volume
    wave *= volume
    
    return wave

def play_beep(name, frequency=800, duration=0.2, waveform='sine', volume=0.3, **kwargs):
    """Play a beep and display info"""
    print(f"\n▶ Playing: {name}")
    print(f"   Frequency: {frequency} Hz | Duration: {duration}s | Waveform: {waveform} | Volume: {volume}")
    
    tone = generate_tone(frequency=frequency, duration=duration, waveform=waveform, volume=volume, **kwargs)
    sd.play(tone, samplerate=44100)
    sd.wait()

def main():
    print("=" * 70)
    print("QuietBox Beep Sound Tester - 30 Options")
    print("=" * 70)
    print("\nTesting different beep sounds. Each beep will play automatically.")
    print("Press Ctrl+C to stop at any time.\n")
    
    time.sleep(1)
    
    # Basic sine waves - various frequencies and volumes
    play_beep("1. Original (Current)", frequency=800, duration=0.2, waveform='sine', volume=0.3)
    time.sleep(0.5)
    
    play_beep("2. Softer Sine", frequency=800, duration=0.2, waveform='sine', volume=0.2)
    time.sleep(0.5)
    
    play_beep("3. Lower Frequency", frequency=600, duration=0.2, waveform='sine', volume=0.3)
    time.sleep(0.5)
    
    play_beep("4. Gentle Higher Pitch", frequency=1000, duration=0.2, waveform='sine', volume=0.25)
    time.sleep(0.5)
    
    play_beep("5. Very Soft Sine", frequency=800, duration=0.2, waveform='sine', volume=0.15)
    time.sleep(0.5)
    
    play_beep("6. Deep Bass Tone", frequency=400, duration=0.2, waveform='sine', volume=0.3)
    time.sleep(0.5)
    
    play_beep("7. Bright High Tone", frequency=1200, duration=0.15, waveform='sine', volume=0.2)
    time.sleep(0.5)
    
    # Bell and chime variations
    play_beep("8. Bell-like Tone", frequency=880, duration=0.3, waveform='bell', volume=0.3)
    time.sleep(0.5)
    
    play_beep("9. Pleasant Chime", frequency=880, duration=0.25, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    play_beep("10. Short Chime", frequency=880, duration=0.15, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    play_beep("11. Soft Bell (Low)", frequency=660, duration=0.25, waveform='bell', volume=0.25)
    time.sleep(0.5)
    
    play_beep("12. Bright Chime", frequency=1047, duration=0.2, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    play_beep("13. Gentle Bell", frequency=880, duration=0.4, waveform='bell', volume=0.2)
    time.sleep(0.5)
    
    # Triangle waves
    play_beep("14. Soft Triangle Wave", frequency=800, duration=0.2, waveform='triangle', volume=0.25)
    time.sleep(0.5)
    
    play_beep("15. Triangle Chime", frequency=880, duration=0.25, waveform='triangle', volume=0.3)
    time.sleep(0.5)
    
    # Musical notes (single tones)
    play_beep("16. Musical Note A4 (440Hz)", frequency=440, duration=0.2, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    play_beep("17. Musical Note C5 (523Hz)", frequency=523, duration=0.2, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    play_beep("18. Musical Note D5 (587Hz)", frequency=587, duration=0.2, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    play_beep("19. Musical Note E5 (659Hz)", frequency=659, duration=0.2, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    play_beep("20. Musical Note G5 (784Hz)", frequency=784, duration=0.2, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    # Multi-tone chimes
    print("\n▶ Playing: 21. Two-Tone Ascending (E5 → G5)")
    print("   Two ascending notes")
    tone1 = generate_tone(frequency=659, duration=0.15, waveform='chime', volume=0.25)
    tone2 = generate_tone(frequency=784, duration=0.15, waveform='chime', volume=0.25)
    combined = np.concatenate([tone1, np.zeros(int(44100 * 0.05)), tone2])
    sd.play(combined, samplerate=44100)
    sd.wait()
    time.sleep(0.5)
    
    print("\n▶ Playing: 22. Three-Tone Ascending (C5 → E5 → G5)")
    print("   Three ascending notes")
    tone1 = generate_tone(frequency=523, duration=0.12, waveform='chime', volume=0.25)
    tone2 = generate_tone(frequency=659, duration=0.12, waveform='chime', volume=0.25)
    tone3 = generate_tone(frequency=784, duration=0.12, waveform='chime', volume=0.25)
    combined = np.concatenate([
        tone1, np.zeros(int(44100 * 0.04)),
        tone2, np.zeros(int(44100 * 0.04)),
        tone3
    ])
    sd.play(combined, samplerate=44100)
    sd.wait()
    time.sleep(0.5)
    
    print("\n▶ Playing: 23. Two-Tone Descending (G5 → E5)")
    print("   Two descending notes")
    tone1 = generate_tone(frequency=784, duration=0.15, waveform='chime', volume=0.25)
    tone2 = generate_tone(frequency=659, duration=0.15, waveform='chime', volume=0.25)
    combined = np.concatenate([tone1, np.zeros(int(44100 * 0.05)), tone2])
    sd.play(combined, samplerate=44100)
    sd.wait()
    time.sleep(0.5)
    
    print("\n▶ Playing: 24. Gentle Two-Tone (D5 → G5)")
    print("   Pleasant interval")
    tone1 = generate_tone(frequency=587, duration=0.15, waveform='chime', volume=0.22)
    tone2 = generate_tone(frequency=784, duration=0.15, waveform='chime', volume=0.22)
    combined = np.concatenate([tone1, np.zeros(int(44100 * 0.06)), tone2])
    sd.play(combined, samplerate=44100)
    sd.wait()
    time.sleep(0.5)
    
    print("\n▶ Playing: 25. Four-Tone Sequence (C5 → D5 → E5 → G5)")
    print("   Musical scale")
    tones = []
    for freq, dur in [(523, 0.1), (587, 0.1), (659, 0.1), (784, 0.15)]:
        tones.append(generate_tone(frequency=freq, duration=dur, waveform='chime', volume=0.2))
        tones.append(np.zeros(int(44100 * 0.03)))
    combined = np.concatenate(tones[:-1])  # Remove last pause
    sd.play(combined, samplerate=44100)
    sd.wait()
    time.sleep(0.5)
    
    # Variations with different durations
    play_beep("26. Quick Chime", frequency=880, duration=0.1, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    play_beep("27. Medium Chime", frequency=880, duration=0.3, waveform='chime', volume=0.3)
    time.sleep(0.5)
    
    play_beep("28. Long Gentle Chime", frequency=880, duration=0.5, waveform='chime', volume=0.2)
    time.sleep(0.5)
    
    # Special combinations
    print("\n▶ Playing: 29. Soft Double Chime")
    print("   Same note twice, soft")
    tone = generate_tone(frequency=880, duration=0.12, waveform='chime', volume=0.2)
    combined = np.concatenate([tone, np.zeros(int(44100 * 0.08)), tone])
    sd.play(combined, samplerate=44100)
    sd.wait()
    time.sleep(0.5)
    
    print("\n▶ Playing: 30. Pleasant Triad Chime (C-E-G)")
    print("   Three notes together")
    tone1 = generate_tone(frequency=523, duration=0.25, waveform='chime', volume=0.18)  # C5
    tone2 = generate_tone(frequency=659, duration=0.25, waveform='chime', volume=0.18)  # E5
    tone3 = generate_tone(frequency=784, duration=0.25, waveform='chime', volume=0.18)  # G5
    # Combine them (play simultaneously)
    max_len = max(len(tone1), len(tone2), len(tone3))
    tone1 = np.pad(tone1, (0, max_len - len(tone1)), 'constant')
    tone2 = np.pad(tone2, (0, max_len - len(tone2)), 'constant')
    tone3 = np.pad(tone3, (0, max_len - len(tone3)), 'constant')
    combined = (tone1 + tone2 + tone3) / 3  # Average to avoid clipping
    sd.play(combined, samplerate=44100)
    sd.wait()
    time.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("All 30 beeps played!")
    print("=" * 70)
    print("\nWhich one did you like? Note the number and let me know!")
    print("\nTo use a beep in main.py, I'll update the play_ready_sound() function with:")
    print("  - frequency")
    print("  - duration")
    print("  - waveform ('sine', 'bell', 'chime', 'triangle', etc.)")
    print("  - volume")
    print("  - any multi-tone pattern if applicable")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"\n\nError: {e}")

