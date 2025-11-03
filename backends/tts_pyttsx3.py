import pyttsx3

class TTSLocal:
    def __init__(self, rate=180, voice_id=None):
        print("[DEBUG] Initializing TTS engine...")
        self.engine = pyttsx3.init()
        
        # Try to find a better voice (prefer female or non-robotic voices)
        voices = self.engine.getProperty('voices')
        
        # Prefer English US/UK voices that might sound better
        preferred_voices = [
            'en-us', 'en-gb', 'en-gb-x-rp',  # English voices
            'en-gb-x-gbcwmd', 'en-gb-x-gbclan'  # Regional variations
        ]
        
        selected_voice = None
        if voice_id is not None:
            # Use specified voice
            try:
                self.engine.setProperty('voice', voices[voice_id].id)
                selected_voice = voices[voice_id].name
            except:
                pass
        
        # Auto-select best available voice
        if selected_voice is None:
            for voice in voices:
                voice_lang = voice.languages[0] if voice.languages else ''
                if any(pref in voice_lang.lower() for pref in preferred_voices):
                    self.engine.setProperty('voice', voice.id)
                    selected_voice = voice.name
                    break
            
            # Fallback to any voice if preferred not found
            if selected_voice is None and len(voices) > 0:
                self.engine.setProperty('voice', voices[0].id)
                selected_voice = voices[0].name
        
        # Set properties for better quality
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', 1.0)  # Max volume
        
        print(f"[DEBUG] TTS engine ready. Voice: {selected_voice}")

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
