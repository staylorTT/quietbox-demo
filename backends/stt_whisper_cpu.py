from faster_whisper import WhisperModel

class STTWhisperCPU:
    def __init__(self, model_size="small"):
        print(f"[DEBUG] Loading Whisper model ({model_size}) - this may take a moment...")
        # device="cpu" and compute_type="int8" keeps it fast on CPU
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"[DEBUG] Whisper model loaded successfully.")

    def transcribe(self, wav_path):
        segments, info = self.model.transcribe(wav_path, beam_size=1)
        text = "".join([seg.text for seg in segments]).strip()
        if not text:
            print(f"[DEBUG] Whisper returned empty transcription. Language detected: {info.language if hasattr(info, 'language') else 'unknown'}")
        return text
