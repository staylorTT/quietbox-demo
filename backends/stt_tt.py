import requests
import base64
import sys


class STTTenstorrent:
    def __init__(self, model_name="openai/whisper-large-v3", base_url="http://localhost:8001/"):
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }

    def wav_to_base64(self, file_path: str) -> str:
        """
        Reads a WAV file and converts it to a base64 encoded string.

        Args:
            file_path: The path to the .wav file.

        Returns:
            A base64 encoded string of the audio file, or an empty
            string if an error occurred.
        """
        try:
            with open(file_path, "rb") as wav_file:
                # 1. Read the entire binary content of the file
                audio_binary = wav_file.read()
                
                # 2. Encode the binary content into base64
                base64_bytes = base64.b64encode(audio_binary)
                
                # 3. Decode the base64 bytes to a UTF-8 string
                base64_string = base64_bytes.decode('utf-8')
                
                return base64_string
                
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.", file=sys.stderr)
            return ""
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)
            return ""

    def respond(self, wav_path):
        # turn wav into base64 encoded string
        audio_base64 = self.wav_to_base64(wav_path)

        # construct payload
        payload = {
            "file": audio_base64,
            "stream": False,
            "is_preprocessing_enabled": False
        }

        # generate response
        response = requests.post(f"{self.base_url}/audio/transcriptions", json=payload, headers=self.headers, timeout=90)
        response.raise_for_status()

        # parse transcription
        transcription = response.json().get("text", "")

        return transcription
    
    def transcribe(self, wav_path):
        """
        Alias for respond() to match interface used by wakeword_whisper and main.py
        """
        return self.respond(wav_path)