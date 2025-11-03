"""
Interruptible TTS wrapper
Allows TTS playback to be interrupted when new speech is detected
"""
import threading
import time
import numpy as np
import sounddevice as sd

class InterruptibleTTS:
    def __init__(self, tts_engine):
        """
        Wrap a TTS engine to make it interruptible
        
        Args:
            tts_engine: The underlying TTS engine (TTSLocal from tts_coqui or tts_pyttsx3)
        """
        self.tts_engine = tts_engine
        self.is_speaking = False
        self.stop_playback = threading.Event()
        self.playback_thread = None
        
    def speak(self, text, interrupt_check_callback=None):
        """
        Speak text, but allow interruption
        
        Args:
            text: Text to speak
            interrupt_check_callback: Optional callback function that returns True if should interrupt
                                      Called periodically during playback
        """
        if not text or not text.strip():
            return
        
        # If we have the underlying speak method that can be made non-blocking
        if hasattr(self.tts_engine, 'speak'):
            # For Coqui TTS, we need to handle it specially
            if hasattr(self.tts_engine, 'tts'):  # Coqui TTS
                self._speak_interruptible_coqui(text, interrupt_check_callback)
            else:  # pyttsx3 or other
                self._speak_interruptible_pyttsx3(text, interrupt_check_callback)
    
    def _speak_interruptible_coqui(self, text, interrupt_check_callback):
        """Handle Coqui TTS with interruption"""
        import tempfile
        import os
        import wave
        from TTS.api import TTS
        
        try:
            # Generate audio file
            temp_dir = tempfile.gettempdir()
            wav_file = os.path.join(temp_dir, f"tts_output_{os.getpid()}.wav")
            
            # Synthesize text to audio file
            self.tts_engine.tts.tts_to_file(text=text, file_path=wav_file)
            
            # Read audio
            if os.path.exists(wav_file):
                with wave.open(wav_file, 'rb') as wf:
                    frames = wf.getnframes()
                    samplerate = wf.getframerate()
                    audio_data = np.frombuffer(wf.readframes(frames), dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Play with interruption checking
                self._play_with_interruption(audio_data, samplerate, interrupt_check_callback)
                
                # Clean up
                try:
                    os.remove(wav_file)
                except:
                    pass
        except Exception as e:
            print(f"[ERROR] Interruptible TTS failed: {e}")
            # Fallback to regular TTS
            self.tts_engine.speak(text)
    
    def _speak_interruptible_pyttsx3(self, text, interrupt_check_callback):
        """Handle pyttsx3 TTS - can't easily interrupt, so just call normally"""
        # pyttsx3 doesn't easily support interruption, so we'll just call it
        # For better interruption, user should use Coqui TTS
        self.tts_engine.speak(text)
    
    def _play_with_interruption(self, audio_data, samplerate, interrupt_check_callback):
        """Play audio with periodic interruption checks"""
        self.is_speaking = True
        self.stop_playback.clear()
        
        chunk_size = int(samplerate * 0.1)  # 100ms chunks for interrupt checking
        chunks = len(audio_data) // chunk_size
        remaining_samples = len(audio_data) % chunk_size
        
        try:
            # Start playing entire audio, but check for interruption periodically
            sd.play(audio_data, samplerate=samplerate)
            
            # Monitor playback progress and check for interruption
            check_interval = 0.1  # Check every 100ms
            total_duration = len(audio_data) / samplerate
            
            elapsed = 0.0
            while elapsed < total_duration:
                # Check if should stop
                if self.stop_playback.is_set():
                    print("\n[INTERRUPT] TTS playback stopped by user")
                    sd.stop()
                    break
                
                # Check interrupt callback if provided
                if interrupt_check_callback and interrupt_check_callback():
                    print("\n[INTERRUPT] TTS playback interrupted by speech")
                    self.stop_playback.set()
                    sd.stop()
                    break
                
                # Check if playback finished
                try:
                    # Try to check stream status
                    current_stream = sd.get_stream()
                    if current_stream is None or not current_stream.active:
                        break
                except:
                    # If we can't check, just continue
                    pass
                
                time.sleep(check_interval)
                elapsed += check_interval
            
            # Wait for remaining playback (with final check)
            if not self.stop_playback.is_set():
                # Final wait, but check one more time
                try:
                    sd.wait()
                except:
                    pass
        
        except Exception as e:
            print(f"[ERROR] Playback error: {e}")
            sd.stop()
        finally:
            self.is_speaking = False
    
    def interrupt(self):
        """Manually interrupt TTS playback"""
        if self.is_speaking:
            self.stop_playback.set()
            sd.stop()
            print("\n[INTERRUPT] TTS manually interrupted")

