"""
Speech Monitor for TTS Interruption
Continuously monitors microphone to detect when user starts speaking
"""
import numpy as np
import sounddevice as sd
import threading
import queue
import time

class SpeechMonitor:
    def __init__(self, device=None, samplerate=16000, threshold=0.001):
        """
        Monitor microphone for speech activity
        
        Args:
            device: Audio device index
            samplerate: Sample rate
            threshold: RMS threshold to consider as speech
        """
        self.device = device
        self.samplerate = samplerate
        self.threshold = threshold
        self.is_monitoring = False
        self.q = queue.Queue()
        self.monitor_thread = None
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            pass  # Could log status issues
        self.q.put(indata.copy())
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        consecutive_speech_frames = 0
        
        while self.is_monitoring:
            try:
                block = self.q.get(timeout=0.1)
                
                # Calculate RMS
                audio = block.squeeze()
                if len(audio.shape) > 1:
                    audio = audio.flatten()
                audio_rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
                
                # Check for speech
                if audio_rms > self.threshold:
                    consecutive_speech_frames += 1
                    # Need 2-3 consecutive frames to avoid false positives
                    if consecutive_speech_frames >= 3:
                        self.speech_detected.set()
                        break
                else:
                    consecutive_speech_frames = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WARNING] Speech monitor error: {e}")
                break
    
    def start_monitoring(self):
        """Start monitoring for speech"""
        self.speech_detected = threading.Event()
        self.is_monitoring = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=int(self.samplerate * 0.1),  # 100ms blocks
            dtype='float32',
            callback=self._audio_callback,
            device=self.device
        )
        self.stream.start()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=0.5)
        # Don't clear speech_detected flag here - let caller check it first
    
    def check_interrupt(self):
        """Check if speech was detected (non-blocking)"""
        if hasattr(self, 'speech_detected'):
            return self.speech_detected.is_set()
        return False
    
    def reset(self):
        """Reset the speech detection flag"""
        if hasattr(self, 'speech_detected'):
            self.speech_detected.clear()

