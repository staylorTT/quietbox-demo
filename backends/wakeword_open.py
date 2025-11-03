import queue, numpy as np, sounddevice as sd
from openwakeword.model import Model
import wave
import os
import time
from datetime import datetime

class WakeWordDetector:
    def __init__(self, keyword=None, samplerate=16000, blocksize=1280, threshold=0.000005, device=None):
        self.samplerate = samplerate
        # OpenWakeWord works best with 1280 samples (80ms) or multiples
        if blocksize < 1280:
            blocksize = 1280
        elif blocksize % 1280 != 0:
            blocksize = int((blocksize // 1280 + 1) * 1280)
        self.blocksize = blocksize
        self.threshold = threshold
        self.device = device  # None = default, or specify device index
        # Note: keyword parameter is unused - OpenWakeWord uses pre-trained models
        # Available wake words are: alexa, hey_mycroft, hey_jarvis, hey_rhasspy, timer, weather
        # Use adaptive threshold - trigger on score spikes above baseline
        self.score_history = []
        self.baseline_score = 0.000005  # Typical background noise level
        self.cooldown_frames = 0  # Cooldown counter to prevent rapid re-triggers
        self.cooldown_duration = 50  # Frames to wait after detection (~4 seconds at 16kHz)
        
        # RMS tracking for audio level monitoring and debug recording
        self.rms_history = []
        self.last_rms_spike_time = 0
        self.rms_spike_cooldown = 2.0  # Don't record more than once per 2 seconds
        self.debug_recordings_dir = "debug_recordings"
        os.makedirs(self.debug_recordings_dir, exist_ok=True)
        
        # Check audio devices (silent unless error)
        try:
            default_input = sd.query_devices(kind='input')
            # Only show if there's an issue
        except Exception as e:
            print(f"[ERROR] Audio device check failed: {e}")
        
        # Load OpenWakeWord models
        try:
            self.model = Model(enable_speex_noise_suppression=True)
        except:
            self.model = Model()
        available_words = list(self.model.models.keys())
        if len(self.model.models) == 0:
            print("[ERROR] No wake word models loaded!")
        else:
            print(f"Wake word models loaded: {', '.join(available_words)}")
            print(f"Threshold set to: {self.threshold} (adjust if needed)")
        self.q = queue.Queue()
        self.audio_buffer = []  # Buffer for debug recordings (keep last ~2 seconds)
        self.max_buffer_frames = 25  # ~2 seconds at 1280 samples per frame

    def _record_debug_utterance(self, frame_num):
        """Save buffered audio when RMS spike is detected"""
        try:
            if len(self.audio_buffer) < 5:  # Need at least some audio
                return
                
            # Convert float32 to int16 for WAV
            audio_data = np.concatenate(self.audio_buffer)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.debug_recordings_dir, f"rms_spike_{timestamp}_frame{frame_num}.wav")
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(self.samplerate)
                wf.writeframes(audio_int16.tobytes())
            
            rms_value = np.sqrt(np.mean(audio_data**2))
            print(f"💾 DEBUG: Saved RMS spike recording: {filename} (RMS: {rms_value:.5f}, {len(self.audio_buffer)} frames)")
                
        except Exception as e:
            print(f"[WARNING] Failed to record debug utterance: {e}")

    def _callback(self, indata, frames, time, status):
        if status:  # Only log actual errors
            print(f"[WARNING] Audio status issue: {status}")
        self.q.put(indata.copy())

    def listen(self):
        """Listen for wake word and return when detected"""
        # Verify device selection
        if self.device is not None:
            try:
                devices = sd.query_devices()
                device_name = devices[self.device]['name']
                print(f"[DEBUG] Wake word detector using device {self.device}: {device_name}")
            except Exception as e:
                print(f"[WARNING] Could not verify wake word device: {e}")
        else:
            try:
                default_input = sd.query_devices(kind='input')
                print(f"[DEBUG] Wake word detector using default device: {default_input['name']}")
            except Exception as e:
                print(f"[WARNING] Could not verify default device: {e}")
        
        with sd.InputStream(channels=1, samplerate=self.samplerate, blocksize=self.blocksize,
                            dtype="float32", callback=self._callback, device=self.device):
            print("Listening for wake word... (say: alexa, hey jarvis, hey mycroft, timer, or weather)")
            frame_count = 0
            no_audio_count = 0  # Track frames with zero audio
            
            while True:
                try:
                    block = self.q.get(timeout=1.0)
                    
                    # Cooldown period: ignore detections right after previous detection
                    if self.cooldown_frames > 0:
                        self.cooldown_frames -= 1
                        frame_count += 1
                        continue
                    
                    # Prepare audio for prediction
                    audio = block.squeeze()
                    if len(audio.shape) > 1:
                        audio = audio.flatten()
                    audio = audio.astype(np.float32)
                    
                    # Buffer audio for debug recording (keep last ~2 seconds)
                    self.audio_buffer.append(audio.copy())
                    if len(self.audio_buffer) > self.max_buffer_frames:
                        self.audio_buffer.pop(0)
                    
                    # Call predict directly on each block (like the working test)
                    try:
                        scores = self.model.predict(audio)
                        max_score = max(scores.values()) if scores else 0.0
                        
                        # Track score history for adaptive detection
                        self.score_history.append(max_score)
                        if len(self.score_history) > 20:  # Keep last 20 frames (~1.6 seconds)
                            self.score_history.pop(0)
                        
                        # Check audio level first - require actual audio input
                        audio_rms = np.sqrt(np.mean(audio**2))
                        
                        # Track RMS history for spike detection
                        self.rms_history.append(audio_rms)
                        if len(self.rms_history) > 10:
                            self.rms_history.pop(0)
                        
                        # Track if we're getting any audio at all
                        if audio_rms < 0.0001:  # Very quiet or silent
                            no_audio_count += 1
                        else:
                            no_audio_count = 0
                        
                        # Warn if no audio for a while
                        if no_audio_count > 50:  # ~4 seconds of silence
                            print(f"[WARNING] No audio detected for {no_audio_count} frames. Check device selection!")
                            no_audio_count = 0  # Reset counter
                        
                        # Show RMS levels more prominently (every 10 frames ~800ms)
                        if frame_count % 10 == 0:
                            rms_baseline = np.mean(self.rms_history) if self.rms_history else 0.0
                            rms_max = max(self.rms_history) if self.rms_history else 0.0
                            status = '📢 SPEAKING' if audio_rms > 0.001 else ('🔇 quiet' if audio_rms > 0.0001 else '⚠️ NO AUDIO')
                            print(f"🎤 RMS: {audio_rms:.5f} | baseline: {rms_baseline:.5f} | max: {rms_max:.5f} | {status}")
                        
                        # Debug: Record utterance when RMS spikes (indicates loud audio)
                        if len(self.rms_history) >= 3:
                            rms_baseline = np.mean(self.rms_history[:-1])  # Baseline before current
                            if audio_rms > 0.001 and audio_rms > rms_baseline * 2.0:
                                # RMS spike detected - record debug audio
                                current_time = time.time()
                                if current_time - self.last_rms_spike_time > self.rms_spike_cooldown:
                                    self._record_debug_utterance(frame_count)
                                    self.last_rms_spike_time = current_time
                        
                        # Adaptive detection: trigger on score spikes above baseline
                        if len(self.score_history) >= 10:
                            baseline = np.mean(self.score_history[-10:])  # Recent average
                            baseline_max = max(self.score_history[-10:])  # Recent peak
                            
                            # Require minimum audio level (0.0005 RMS - lower to catch quiet speech) to prevent false positives from noise
                            # AND one of the trigger conditions:
                            # 1. Above absolute threshold, OR
                            # 2. Significantly above baseline (2x), OR  
                            # 3. Above baseline_max (new peak)
                            # Lower thresholds to be more sensitive
                            audio_threshold = 0.0005  # Lower from 0.001
                            score_threshold = max(self.threshold, baseline * 1.5)  # Lower from 2.0x baseline
                            peak_threshold = baseline_max * 1.1  # Lower from 1.2
                            
                            if audio_rms >= audio_threshold and (
                                max_score >= self.threshold or \
                                (baseline > 0 and max_score > score_threshold) or \
                                max_score > peak_threshold
                            ):
                                max_word = max(scores.items(), key=lambda x: x[1])[0] if scores else None
                                print(f"✓ Wake word '{max_word}' detected! (score: {max_score:.6f}, baseline: {baseline:.6f}, audio={audio_rms:.4f})")
                                # Set cooldown to prevent rapid re-triggers
                                self.cooldown_frames = self.cooldown_duration
                                self.score_history.clear()  # Reset history after detection
                                return  # Wake word detected!
                        
                        # Show score spikes for debugging (more frequent)
                        if len(self.score_history) >= 5 and max_score > 0.000005:
                            baseline = np.mean(self.score_history[-5:])
                            if max_score > baseline * 1.2:  # 20% above baseline (more sensitive)
                                max_word = max(scores.items(), key=lambda x: x[1])[0] if scores else None
                                audio_rms = np.sqrt(np.mean(audio**2))
                                print(f"⚠️  Score spike: {max_word}={max_score:.6f} (baseline: {baseline:.6f}, audio: {audio_rms:.5f}) - NOT TRIGGERING YET")
                        
                        # Show detailed status less frequently (every ~3 seconds)
                        if frame_count % 40 == 0:
                            baseline = np.mean(self.score_history[-10:]) if len(self.score_history) >= 10 else 0.0
                            baseline_max = max(self.score_history[-10:]) if len(self.score_history) >= 10 else 0.0
                            max_word = max(scores.items(), key=lambda x: x[1])[0] if scores else None
                            print(f"📊 Status: score={max_score:.6f}, word={max_word}, baseline={baseline:.6f}, max={baseline_max:.6f}, threshold={self.threshold:.6f}")
                            
                    except Exception as e:
                        print(f"[ERROR] Prediction failed: {e}")
                    
                    frame_count += 1
                    
                except queue.Empty:
                    if frame_count == 0:
                        print("[WARNING] No audio received. Check microphone.")
                    break  # Timeout - should not happen in normal operation
