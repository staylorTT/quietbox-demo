import collections, sys, wave, time
import numpy as np, sounddevice as sd, webrtcvad

class Recorder:
    def __init__(self, samplerate=16000, frame_ms=30, aggressiveness=2, max_seconds=15, silence_tail_ms=1200, device=None):
        print("[DEBUG] Setting up recorder with VAD...")
        self.samplerate = samplerate
        self.frame_len = int(samplerate * frame_ms / 1000)
        self.vad = webrtcvad.Vad(aggressiveness)
        self.max_seconds = max_seconds
        self.silence_tail_ms = silence_tail_ms  # Store in ms for use in record_to_wav
        self.silence_tail_frames = int(silence_tail_ms / (frame_ms))
        self.device = device  # Audio device index (None = default)
        if device is not None:
            try:
                devices = sd.query_devices()
                device_name = devices[device]['name']
                print(f"[DEBUG] Recorder using device {device}: {device_name}")
            except:
                print(f"[DEBUG] Recorder using device {device}")
        else:
            try:
                default_input = sd.query_devices(kind='input')
                print(f"[DEBUG] Recorder using default device: {default_input['name']}")
            except:
                print("[DEBUG] Recorder using default device")
        print("[DEBUG] Recorder ready.")

    def _frame_gen(self):
        with sd.InputStream(channels=1, samplerate=self.samplerate, dtype="int16", blocksize=self.frame_len, device=self.device) as stream:
            while True:
                buf, _ = stream.read(self.frame_len)
                yield bytes(buf)

    def record_to_wav(self, out_path="utterance.wav", min_seconds=0.5):
        print("[DEBUG] Starting VAD recording...")
        frames = []
        voiced_window = collections.deque(maxlen=self.silence_tail_frames)
        start = time.time()
        frames_with_speech = 0
        min_frames = int(self.samplerate * min_seconds / self.frame_len)  # Minimum frames to record
        last_speech_time = start  # Track when we last detected speech
        speech_detected = False  # Track if we've detected ANY speech
        
        for f in self._frame_gen():
            is_voiced = self.vad.is_speech(f, sample_rate=self.samplerate)
            frames.append(f)
            if is_voiced:
                frames_with_speech += 1
                last_speech_time = time.time()  # Update last speech time
                speech_detected = True  # Mark that we've detected speech
            voiced_window.append(1 if is_voiced else 0)

            elapsed = time.time() - start
            
            # Safety: if no speech detected after 5 seconds, stop anyway (don't wait forever)
            if not speech_detected and elapsed > 5.0:
                print(f"[DEBUG] Recording stopped: no speech detected after 5s ({len(frames)} frames)")
                break
            
            # Only check for silence-based stopping after minimum time AND speech was detected
            if len(frames) >= min_frames and speech_detected:
                # Stop if we've had sufficient silence AFTER detecting speech
                silence_duration = time.time() - last_speech_time
                if silence_duration >= (self.silence_tail_ms / 1000.0):
                    print(f"[DEBUG] Recording stopped: {silence_duration:.2f}s silence after speech ({len(frames)} frames, {frames_with_speech} with speech)")
                    break
            
            # Timeout after max seconds
            if elapsed > self.max_seconds:
                print(f"[DEBUG] Recording stopped: timeout ({len(frames)} frames, {frames_with_speech} with speech)")
                break

        # Calculate audio stats
        if frames:
            audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
            audio_rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            audio_duration = len(frames) * (self.frame_len / self.samplerate)
            print(f"[DEBUG] Recorded: {audio_duration:.2f}s, {len(frames)} frames, RMS={audio_rms:.4f}, speech_frames={frames_with_speech}")
            
            if audio_rms < 100:  # Very low audio level
                print("[WARNING] Audio level is very low - may be silent or wrong device!")
            if frames_with_speech == 0:
                print("[WARNING] No speech frames detected by VAD!")
        else:
            print("[ERROR] No frames recorded!")
            audio_rms = 0

        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.samplerate)
            wf.writeframes(b"".join(frames))
        print(f"[DEBUG] Saved recording to {out_path}")
        return out_path
