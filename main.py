from backends.wakeword_open import WakeWordDetector
from backends.record_vad import Recorder
from backends.stt_whisper_cpu import STTWhisperCPU
from backends.llm_hf_cpu import ResponderHFCPU
from backends.tts_pyttsx3 import TTSLocal
from datetime import datetime
import os
import numpy as np
import sounddevice as sd
# from backends.stt_tt import STTTenstorrent
from backends.llm_tt import ResponderTenstorrent

def play_ready_sound(frequency=800, duration=0.2, volume=0.3):
    """Play a simple beep/ding sound to indicate ready state"""
    try:
        samplerate = 22050
        t = np.linspace(0, duration, int(samplerate * duration))
        # Create a pleasant two-tone ding
        tone1 = np.sin(2 * np.pi * frequency * t) * volume
        tone2 = np.sin(2 * np.pi * (frequency * 1.5) * t[:len(t)//2]) * volume * 0.7
        # Add a fade in/out to avoid clicks
        fade_samples = int(samplerate * 0.01)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone1[:fade_samples] *= fade_in
        tone1[-fade_samples:] *= fade_out
        
        # Combine tones
        audio = tone1
        if len(tone2) < len(audio):
            audio[:len(tone2)] += tone2
        
        sd.play(audio, samplerate=samplerate)
        sd.wait()
    except Exception as e:
        # Silent failure - don't interrupt if sound fails
        pass

def build_pipeline(mode="cpu"):
    print("Initializing QuietBox pipeline...")
    # Swap these two lines later to TT backends once ready
    if mode == "cpu-rest_tt-llm":
        print("Loading Whisper model (STT)...")
        stt = STTWhisperCPU(model_size="small")
        print("Loading Llama-3.1-8B-Instruct model (LLM)...")
        llm = ResponderTenstorrent(model_name="meta-llama/Llama-3.1-8B-Instruct")
    elif mode == "cpu":
        print("Loading Whisper model (STT)...")
        stt = STTWhisperCPU(model_size="small")
        print("Loading TinyLlama model (LLM) - this may take a minute...")
        llm = ResponderHFCPU(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    elif mode == "tt":
        # stt = STTTenstorrent(model_artifacts_path="/opt/tt/models/whisper")
        # llm = ResponderTenstorrent(model_artifacts_path="/opt/tt/models/tinyllama")
        raise NotImplementedError("TT mode not wired yet")
    else:
        raise ValueError("mode must be 'cpu' or 'tt'")

    print("Initializing TTS...")
    # Try Coqui TTS first (neural, much more natural), fallback to pyttsx3
    # Voice options:
    # - "tts_models/en/ljspeech/fast_pitch" - Fast with good prosody/naturalness (default)
    # - "tts_models/en/ljspeech/glow-tts" - Fast neural, natural sounding
    # - "tts_models/en/ljspeech/overflow" - Very high quality (slower)
    # - "tts_models/en/vctk/vits" - Multiple speaker voices
    # - "tts_models/en/ljspeech/speedy-speech" - Very fast
    import os
    selected_voice = os.environ.get("QUIETBOX_VOICE", "tts_models/en/ljspeech/fast_pitch")
    
    try:
        from backends.tts_coqui import TTSLocal as TTSCoqui
        tts = TTSCoqui(voice=selected_voice, use_gpu=False)
        voice_name = selected_voice.split("/")[-1]
        print(f"[INFO] Using Coqui TTS (neural voice: {voice_name}) for natural speech")
    except Exception as e:
        print(f"[INFO] Coqui TTS not available ({e}), using pyttsx3")
        tts = TTSLocal()
    
    # Device selection: Use PulseAudio virtual devices which route correctly
    # PulseAudio handles sample rate conversion and routing to actual hardware
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        
        # Prefer PulseAudio devices (they route to actual hardware properly)
        # Device 9 = 'pulse', Device 10 = 'default' (both are PulseAudio virtual devices)
        pulse_devices = [d for d in devices if d['name'].lower() in ['pulse', 'default'] and d['max_input_channels'] > 0]
        
        # Use PulseAudio default (device 10) which routes to the active mic
        target_device = 10  # PulseAudio 'default' - routes to system default input
        
        # Verify device exists
        if target_device < len(devices) and devices[target_device]['max_input_channels'] > 0:
            device_name = devices[target_device]['name']
            print(f"[DEBUG] Using PulseAudio default device {target_device}: {device_name}")
            print(f"[INFO] This routes to your system's active microphone")
        else:
            # Fallback to device index 10 (default)
            default_input = sd.query_devices(kind='input')
            target_device = default_input['index']
            device_name = default_input['name']
            print(f"[DEBUG] Using default input device {target_device}: {device_name}")
        
        # Initialize wake word detector with PulseAudio default
        wake = WakeWordDetector(device=target_device)
        
    except Exception as e:
        print(f"[WARNING] Device detection failed, using default: {e}")
        target_device = None
        wake = WakeWordDetector()  # Will use system default
    
    # Use the SAME device for recorder
    rec = Recorder(device=target_device)
    print("Pipeline ready!")

    return wake, rec, stt, llm, tts

def run_loop(mode="cpu", use_wake_word=True):
    wake, rec, stt, llm, tts = build_pipeline(mode)
    
    # Create utterances directory if it doesn't exist
    utterances_dir = "utterances"
    os.makedirs(utterances_dir, exist_ok=True)
    print(f"Utterances will be saved to: {utterances_dir}/")
    
    if use_wake_word:
        print("QuietBox ready. Say your wake word to start.")
    else:
        print("QuietBox ready. Press Enter to start listening (or Ctrl+C to exit).")
        print("Note: Wake word detection disabled - using keyboard trigger instead.")
    
    while True:
        # 1) wait for wake word OR keyboard input
        if use_wake_word:
            wake.listen()
            print("Wake word detected.")
        else:
            # Simple keyboard trigger for testing
            try:
                input("Press Enter when ready to speak your question... ")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
        
        # Play chime to indicate ready to record
        play_ready_sound()
        
        # Clear timing feedback with countdown (shorter countdown)
        import time
        print("\n" + "="*60)
        print("🎤 READY TO RECORD")
        print("="*60)
        print("Recording starts in 1 second...")
        time.sleep(0.3)  # Shorter delay
        print("🎙️  SPEAK NOW!")
        print("="*60 + "\n")
        
        # 2) record with VAD - save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_filename = os.path.join(utterances_dir, f"utterance_{timestamp}.wav")
        wav = rec.record_to_wav(wav_filename)
        print(f"✓ Recording stopped. Saved: {wav_filename}")
        
        # Check if recording has meaningful audio before transcribing
        import wave
        duration = 0.0
        try:
            with wave.open(wav, 'rb') as wf:
                frames = wf.getnframes()
                duration = frames / wf.getframerate()
                if duration < 0.3:  # Less than 300ms - likely empty or too short
                    print(f"[SKIP] Recording too short ({duration:.2f}s), skipping transcription")
                    continue
        except Exception as e:
            print(f"[WARNING] Could not check recording duration: {e}")
            # Continue anyway if we can't check
        
        print("🔄 Transcribing audio...")
        # 3) STT
        text = stt.transcribe(wav)
        print(f"📝 You said: {text}")
        if not text:
            # Only speak error if recording was long enough to be valid
            if duration >= 0.3:
                print("⚠️  Transcription empty - asking user to repeat...")
                tts.speak("Sorry, I did not catch that.")
            continue
        
        # 4) LLM response
        print("🤔 Thinking...")
        # Speak "Thinking" audibly
        tts.speak("Thinking")
        reply = llm.respond(text)
        print(f"💬 Assistant: {reply}")
        
        # 5) TTS
        print("🔊 Speaking response...")
        tts.speak(reply)
        print("\n" + "="*60)
        if use_wake_word:
            print("✅ Ready for the next wake word.")
        else:
            print("✅ Ready for next question.")
        print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    # Check for JWT_SECRET if using Tenstorrent backend
    # Load from .env file if it exists
    import os
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    # Parse command line arguments
    # --no-wake-word: Disable wake word detection
    # --voice MODEL: Use specific TTS voice model (e.g., --voice glow-tts)
    use_wake_word = "--no-wake-word" not in sys.argv
    
    # Voice selection
    voice_map = {
        "fast_pitch": "tts_models/en/ljspeech/fast_pitch",
        "glow-tts": "tts_models/en/ljspeech/glow-tts",
        "overflow": "tts_models/en/ljspeech/overflow",
        "tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
        "speedy": "tts_models/en/ljspeech/speedy-speech",
        "vits": "tts_models/en/vctk/vits",
    }
    
    selected_voice = "tts_models/en/ljspeech/fast_pitch"  # Default
    for arg in sys.argv:
        if arg.startswith("--voice="):
            voice_name = arg.split("=", 1)[1]
            if voice_name in voice_map:
                selected_voice = voice_map[voice_name]
                print(f"[INFO] Using voice: {voice_name} ({selected_voice})")
            else:
                print(f"[WARNING] Unknown voice '{voice_name}'. Available: {list(voice_map.keys())}")
    
    # Store for use in build_pipeline
    import os
    os.environ["QUIETBOX_VOICE"] = selected_voice
    
    run_loop(mode="cpu-rest_tt-llm", use_wake_word=use_wake_word)
