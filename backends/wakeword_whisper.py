"""
Whisper-based Wake Word Detection
Uses continuous Whisper transcription to detect wake phrases

This is more flexible than keyword detection - you can use any phrase you want!
"""
import numpy as np
import sounddevice as sd
import time
import tempfile
import wave
import os

class WakeWordDetector:
    def __init__(self, wake_phrases=None, samplerate=16000, chunk_duration=2.0, 
                 device=None, stt_model_size="base"):
        """
        Initialize Whisper-based wake word detector
        
        Args:
            wake_phrases: List of phrases to detect (e.g., ["hey assistant", "okay computer"])
                         Default: ["hey quietbox", "okay quietbox", "hey assistant"]
            samplerate: Audio sample rate
            chunk_duration: How long to record before checking (seconds)
            device: Audio device index
            stt_model_size: Whisper model size (tiny/base/small for speed, larger for accuracy)
        """
        self.samplerate = samplerate
        self.chunk_duration = chunk_duration
        self.device = device
        self.chunk_samples = int(samplerate * chunk_duration)
        
        # Default wake phrases - user can customize
        if wake_phrases is None:
            self.wake_phrases = [
                "hey quiet box",
                "okay quiet box", 
                "hey assistant",
                "okay assistant",
                "hey computer",
                "okay computer",
                "hey quietbox",
                "okay quietbox", 
            ]
        else:
            self.wake_phrases = [p.lower() for p in wake_phrases]
        
        print(f"[DEBUG] Initializing Whisper wake word detector...")
        print(f"[DEBUG] Wake phrases: {', '.join(self.wake_phrases)}")
        print(f"[DEBUG] Using Whisper model: {stt_model_size}")
        
        # Load Whisper model (use smaller model for speed)
        # Import here to avoid circular dependency
        from .stt_whisper_cpu import STTWhisperCPU
        self.stt = STTWhisperCPU(model_size=stt_model_size)
        print(f"[DEBUG] Whisper wake word detector ready")
    
    def _record_chunk(self):
        """Record a chunk of audio"""
        recording = sd.rec(int(self.chunk_samples), 
                          samplerate=self.samplerate, 
                          channels=1, 
                          dtype='float32',
                          device=self.device)
        sd.wait()
        return recording
    
    def _save_temp_audio(self, audio_data):
        """Save audio to temporary WAV file for Whisper"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        # Convert float32 to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_int16.tobytes())
        
        return temp_path
    
    def _check_wake_phrase(self, text):
        """Check if transcribed text contains any wake phrase"""
        if not text:
            return False
        
        # Normalize text: lowercase, remove punctuation for better matching
        text_lower = text.lower().strip()
        # Remove common punctuation
        import string
        text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
        
        # Log what we're checking
        print(f"[DEBUG] Checking wake phrase: original='{text}', cleaned='{text_clean}'")
        print(f"[DEBUG] Looking for phrases: {self.wake_phrases}")
        
        # Check for exact phrase matches in cleaned text
        for phrase in self.wake_phrases:
            phrase_clean = phrase.lower().translate(str.maketrans('', '', string.punctuation))
            if phrase_clean in text_clean:
                print(f"[DEBUG] ✓ Matched phrase '{phrase}' in '{text_clean}'")
                return True
        
        # Also check word-by-word matching (more forgiving for transcription errors)
        text_words = text_clean.split()
        for phrase in self.wake_phrases:
            phrase_clean = phrase.lower().translate(str.maketrans('', '', string.punctuation))
            phrase_words = phrase_clean.split()
            
            # Check if all words in phrase appear in order in the text
            if len(phrase_words) <= len(text_words):
                # Try to find phrase words in sequence
                word_idx = 0
                matched = True
                for phrase_word in phrase_words:
                    found = False
                    # Search from current position
                    for i in range(word_idx, len(text_words)):
                        # Allow fuzzy matching (word contains or is contained by phrase word)
                        if phrase_word in text_words[i] or text_words[i] in phrase_word:
                            word_idx = i + 1
                            found = True
                            break
                    if not found:
                        matched = False
                        break
                
                if matched:
                    print(f"[DEBUG] ✓ Matched phrase '{phrase}' (word-by-word) in '{text_clean}'")
                    return True
        
        # Fallback: check for partial matches (e.g., "hey" or "okay" at start)
        if text_words:
            first_word = text_words[0]
            if first_word in ["hey", "okay", "ok"]:
                # If followed by common assistant words
                if len(text_words) > 1:
                    second_word = text_words[1]
                    if second_word in ["assistant", "computer", "quiet", "box", "quietbox"]:
                        print(f"[DEBUG] ✓ Matched partial phrase '{first_word} {second_word}'")
                        return True
        
        print(f"[DEBUG] ✗ No wake phrase match found")
        return False
    
    def listen(self):
        """Listen continuously until wake phrase is detected"""
        print(f"🎤 Listening for wake phrase... (say: {', '.join(self.wake_phrases[:3])}...)")
        
        frame_count = 0
        rms_history = []
        
        while True:
            try:
                # Record a chunk
                audio_chunk = self._record_chunk()
                
                # Calculate RMS for audio level monitoring
                audio_flat = audio_chunk.flatten()
                audio_rms = np.sqrt(np.mean(audio_flat**2))
                rms_history.append(audio_rms)
                if len(rms_history) > 10:
                    rms_history.pop(0)
                
                # Show RMS status every chunk (helps debug mic input)
                rms_baseline = np.mean(rms_history) if rms_history else 0.0
                rms_max = max(rms_history) if rms_history else 0.0
                status = '📢 SPEAKING' if audio_rms > 0.001 else ('🔇 quiet' if audio_rms > 0.0001 else '⚠️ NO AUDIO')
                print(f"🎤 RMS: {audio_rms:.5f} | baseline: {rms_baseline:.5f} | max: {rms_max:.5f} | {status}")
                
                # Warn if no audio detected for a while
                if audio_rms < 0.0001 and frame_count > 5:
                    print(f"[WARNING] Very low audio (RMS={audio_rms:.5f}). Check microphone connection!")
                
                # Save to temp file
                temp_wav = self._save_temp_audio(audio_flat)
                
                # Transcribe with Whisper
                try:
                    text = self.stt.transcribe(temp_wav)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                    
                    # Check if wake phrase detected
                    if text and self._check_wake_phrase(text):
                        print(f"✓ Wake phrase detected! Heard: '{text}'")
                        return
                    
                    # Show transcription status
                    if text:
                        text_preview = text[:50] + "..." if len(text) > 50 else text
                        print(f"💭 Transcribed: '{text_preview}' (no wake phrase)")
                    else:
                        print(f"🔇 No speech transcribed (RMS was {audio_rms:.5f})")
                    
                except Exception as e:
                    print(f"[WARNING] Transcription error: {e}")
                    # Clean up temp file on error
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                
                frame_count += 1
                
            except KeyboardInterrupt:
                print("\n[INFO] Wake word detection interrupted")
                raise
            except Exception as e:
                print(f"[ERROR] Wake word detection error: {e}")
                time.sleep(0.5)  # Brief pause before retry

