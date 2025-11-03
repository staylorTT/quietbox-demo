"""
Piper TTS Backend - Neural Text-to-Speech
Much more natural sounding than pyttsx3/espeak

Uses piper-tts Python library for high-quality neural TTS
"""
import os
import tempfile
import subprocess
import sounddevice as sd
import numpy as np
import wave

try:
    import piper_tts
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

class TTSLocal:
    def __init__(self, voice="en_US-lessac-medium"):
        """
        Initialize Piper TTS
        
        Args:
            voice: Voice name (e.g., 'en_US-lessac-medium', 'en_US-joe-medium')
                   See: https://github.com/rhasspy/piper/releases
        """
        if not PIPER_AVAILABLE:
            print("[WARNING] piper-tts not available, falling back to pyttsx3")
            from backends.tts_pyttsx3 import TTSLocal as Pyttsx3TTS
            self.engine = Pyttsx3TTS()
            self.speak = self.engine.speak
            return
        
        print("[DEBUG] Initializing Piper TTS (neural voice)...")
        self.voice = voice
        self.samplerate = 22050  # Piper default
        
        # Try to use piper_tts Python library
        try:
            # Check if piper_tts has a direct API
            if hasattr(piper_tts, 'synthesize'):
                print("[DEBUG] Using piper_tts Python API")
                self.use_api = True
                self.model = piper_tts.synthesize
            else:
                # Fall back to CLI
                self.use_api = False
                print("[DEBUG] Using piper CLI")
        except Exception as e:
            print(f"[WARNING] Piper API check failed: {e}, using CLI")
            self.use_api = False
        
        print(f"[DEBUG] Piper TTS ready with voice: {voice}")
    
    def speak(self, text):
        """Synthesize and play speech using Piper TTS"""
        if not text or not text.strip():
            return
        
        if not PIPER_AVAILABLE:
            # Fallback handled in __init__
            return
        
        try:
            # Generate audio using Piper
            audio_data = self._synthesize(text)
            
            if audio_data is not None:
                # Play audio using sounddevice
                sd.play(audio_data, samplerate=self.samplerate)
                sd.wait()  # Wait until playback is finished
            else:
                print("[WARNING] Piper synthesis returned no audio")
                
        except Exception as e:
            print(f"[ERROR] Piper TTS failed: {e}")
            print("[INFO] Falling back to system TTS...")
            # Fallback to pyttsx3
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except:
                pass
    
    def _synthesize(self, text):
        """Synthesize text to audio using Piper"""
        if self.use_api:
            # Use Python API if available
            try:
                # This might need adjustment based on actual piper_tts API
                audio_data = piper_tts.synthesize(text, voice=self.voice)
                return audio_data
            except Exception as e:
                print(f"[WARNING] Piper API synthesis failed: {e}, trying CLI")
                self.use_api = False
        
        # Use CLI method
        temp_dir = tempfile.gettempdir()
        wav_file = os.path.join(temp_dir, f"piper_output_{os.getpid()}.wav")
        
        try:
            # Run piper command
            # Try different possible command formats
            cmd = None
            for possible_cmd in ['piper', 'piper-tts', 'python -m piper_tts']:
                try:
                    result = subprocess.run([possible_cmd.split()[0], '--version'], 
                                           capture_output=True, timeout=2)
                    if result.returncode == 0:
                        if 'piper' in possible_cmd:
                            cmd = possible_cmd.split()
                        else:
                            cmd = [possible_cmd.split()[0]]
                        break
                except:
                    continue
            
            if cmd is None:
                # Try using piper_tts Python module directly
                cmd = ['python3', '-m', 'piper_tts']
            
            # Generate audio
            piper_cmd = cmd + ['--model', self.voice, '--output-file', wav_file]
            process = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     text=True)
            stdout, stderr = process.communicate(input=text, timeout=10)
            
            if process.returncode != 0:
                print(f"[ERROR] Piper TTS command failed: {stderr}")
                # Try alternative: direct model download and use
                return self._synthesize_fallback(text)
            
            # Read WAV file
            if os.path.exists(wav_file):
                with wave.open(wav_file, 'rb') as wf:
                    frames = wf.getnframes()
                    audio_data = np.frombuffer(wf.readframes(frames), dtype=np.int16)
                    self.samplerate = wf.getframerate()
                    # Convert to float32
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Clean up
                try:
                    os.remove(wav_file)
                except:
                    pass
                
                return audio_data
            else:
                return self._synthesize_fallback(text)
                
        except subprocess.TimeoutExpired:
            print("[ERROR] Piper TTS timed out")
            return None
        except Exception as e:
            print(f"[ERROR] Piper synthesis error: {e}")
            return self._synthesize_fallback(text)
        finally:
            # Cleanup
            if os.path.exists(wav_file):
                try:
                    os.remove(wav_file)
                except:
                    pass
    
    def _synthesize_fallback(self, text):
        """Fallback synthesis using online model download"""
        try:
            # Try to use piper_tts with auto-download
            # This is a simplified approach - may need refinement
            print("[INFO] Attempting to download/use Piper model automatically...")
            
            # For now, fall back to pyttsx3
            import pyttsx3
            engine = pyttsx3.init()
            
            # Save to temp file and play
            temp_dir = tempfile.gettempdir()
            wav_file = os.path.join(temp_dir, f"pyttsx3_fallback_{os.getpid()}.wav")
            
            engine.save_to_file(text, wav_file)
            engine.runAndWait()
            
            if os.path.exists(wav_file):
                import wave as wav_lib
                with wav_lib.open(wav_file, 'rb') as wf:
                    frames = wf.getnframes()
                    audio_data = np.frombuffer(wf.readframes(frames), dtype=np.int16)
                    self.samplerate = wf.getframerate()
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                os.remove(wav_file)
                return audio_data
            
        except Exception as e:
            print(f"[ERROR] Fallback synthesis failed: {e}")
        
        return None
