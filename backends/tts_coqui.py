"""
Coqui TTS Backend - High-quality neural Text-to-Speech
Much more natural sounding than pyttsx3/espeak

Uses Coqui TTS for high-quality neural voice synthesis
"""
import sounddevice as sd
import numpy as np
import tempfile
import os

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

class TTSLocal:
    def __init__(self, voice="tts_models/en/ljspeech/fast_pitch", use_gpu=False):
        """
        Initialize Coqui TTS
        
        Args:
            voice: Model name from Coqui TTS
                   Options:
                   - "tts_models/en/ljspeech/fast_pitch" (fast, natural prosody - recommended)
                   - "tts_models/en/ljspeech/glow-tts" (fast, natural)
                   - "tts_models/en/ljspeech/overflow" (very high quality, slower)
                   - "tts_models/en/ljspeech/tacotron2-DDC" (balanced)
                   - "tts_models/en/ljspeech/speedy-speech" (very fast)
                   - "tts_models/en/vctk/vits" (multiple speaker voices)
            use_gpu: Use GPU if available (faster)
        """
        if not COQUI_AVAILABLE:
            print("[WARNING] Coqui TTS not available, falling back to pyttsx3")
            from backends.tts_pyttsx3 import TTSLocal as Pyttsx3TTS
            self.engine = Pyttsx3TTS()
            self.speak = self.engine.speak
            return
        
        print("[DEBUG] Initializing Coqui TTS (neural voice)...")
        self.voice = voice
        self.use_gpu = use_gpu
        
        try:
            # Initialize TTS model
            self.tts = TTS(model_name=voice, progress_bar=False, gpu=use_gpu)
            print(f"[DEBUG] Coqui TTS ready with model: {voice}")
        except Exception as e:
            print(f"[ERROR] Failed to load Coqui TTS model: {e}")
            print("[INFO] Falling back to pyttsx3")
            from backends.tts_pyttsx3 import TTSLocal as Pyttsx3TTS
            self.engine = Pyttsx3TTS()
            self.speak = self.engine.speak
    
    def speak(self, text):
        """Synthesize and play speech using Coqui TTS"""
        if not text or not text.strip():
            return
        
        if not COQUI_AVAILABLE or not hasattr(self, 'tts'):
            # Fallback handled in __init__
            return
        
        try:
            # Generate audio
            temp_dir = tempfile.gettempdir()
            wav_file = os.path.join(temp_dir, f"coqui_output_{os.getpid()}.wav")
            
            # Synthesize text to audio file
            self.tts.tts_to_file(text=text, file_path=wav_file)
            
            # Read and play audio
            if os.path.exists(wav_file):
                import wave
                with wave.open(wav_file, 'rb') as wf:
                    frames = wf.getnframes()
                    samplerate = wf.getframerate()
                    audio_data = np.frombuffer(wf.readframes(frames), dtype=np.int16)
                    # Convert to float32 and normalize
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Play audio using sounddevice
                sd.play(audio_data, samplerate=samplerate)
                sd.wait()  # Wait until playback is finished
                
                # Clean up
                try:
                    os.remove(wav_file)
                except:
                    pass
            else:
                print("[WARNING] Coqui TTS did not generate audio file")
                self._fallback_speak(text)
                
        except Exception as e:
            print(f"[ERROR] Coqui TTS failed: {e}")
            self._fallback_speak(text)
    
    def _fallback_speak(self, text):
        """Fallback to pyttsx3"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[ERROR] Fallback TTS also failed: {e}")

