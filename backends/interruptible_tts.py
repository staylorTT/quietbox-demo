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
        
    def speak(self, text, interrupt_check_callback=None, ui_callback=None, ui_process_events=None):
        """
        Speak text, but allow interruption
        
        Args:
            text: Text to speak
            interrupt_check_callback: Optional callback function that returns True if should interrupt
                                      Called periodically during playback
            ui_callback: Optional callback for real-time audio levels (called with RMS value)
        """
        if not text or not text.strip():
            return
        
        # If we have the underlying speak method that can be made non-blocking
        if hasattr(self.tts_engine, 'speak'):
            # For Coqui TTS, we need to handle it specially
            if hasattr(self.tts_engine, 'tts'):  # Coqui TTS
                self._speak_interruptible_coqui(text, interrupt_check_callback, ui_callback, ui_process_events)
            else:  # pyttsx3 or other
                self._speak_interruptible_pyttsx3(text, interrupt_check_callback)
    
    def _speak_interruptible_coqui(self, text, interrupt_check_callback, ui_callback=None, ui_process_events=None):
        """Handle Coqui TTS with interruption and UI updates"""
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
                
                # Play with interruption checking AND UI updates
                self._play_with_interruption(audio_data, samplerate, interrupt_check_callback, ui_callback, ui_process_events)
                
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
    
    def _play_with_interruption(self, audio_data, samplerate, interrupt_check_callback=None, ui_callback=None, ui_process_events=None):
        """Play audio with periodic interruption checks and real-time UI updates"""
        self.is_speaking = True
        self.stop_playback.clear()
        
        # Play entire audio at once for smooth playback, but analyze in chunks for UI
        # This prevents garbled audio while still providing real-time visualization
        try:
            # Start playing the entire audio
            sd.play(audio_data, samplerate=samplerate)
            
            # Monitor playback and send audio levels to UI
            chunk_duration = 0.05  # 50ms chunks for UI analysis
            chunk_size = int(samplerate * chunk_duration)
            total_chunks = len(audio_data) // chunk_size
            check_interval = 0.05  # Check every 50ms
            
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
                
                # Calculate RMS for current playback position (for UI visualization)
                if ui_callback:
                    try:
                        # Estimate current position in audio
                        current_sample = int(elapsed * samplerate)
                        end_sample = min(current_sample + chunk_size, len(audio_data))
                        
                        if current_sample < len(audio_data):
                            current_chunk = audio_data[current_sample:end_sample]
                            if len(current_chunk) > 0:
                                # Calculate RMS - use absolute value to handle negative samples
                                audio_abs = np.abs(current_chunk.astype(np.float32))
                                audio_rms = np.sqrt(np.mean(audio_abs**2))
                                
                                # Only send if RMS is meaningful (above noise floor)
                                if audio_rms > 0.0001:
                                    ui_callback(audio_rms)
                    except Exception as e:
                        print(f"[DEBUG] UI callback error: {e}")
                        pass  # Don't fail if UI callback errors
                
                # Process tkinter events periodically (non-blocking) - needed for root.after() to work
                if ui_process_events:
                    try:
                        ui_process_events()
                    except:
                        pass  # Ignore errors if window is closed
                
                # Check if playback finished
                try:
                    current_stream = sd.get_stream()
                    if current_stream is None or not current_stream.active:
                        break
                except:
                    pass
                
                time.sleep(check_interval)
                elapsed += check_interval
            
            # Wait for any remaining playback
            if not self.stop_playback.is_set():
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

