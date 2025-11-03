# QuietBox Demo

A simple interactive voice assistant demo that demonstrates a complete speech-to-text → LLM → text-to-speech pipeline.

## Overview

QuietBox is a voice assistant that:
1. **Listens** for a wake word (using OpenWakeWord)
2. **Records** your speech with Voice Activity Detection (VAD)
3. **Transcribes** your speech to text (using Whisper)
4. **Generates** a response using an LLM (TinyLlama)
5. **Speaks** the response back to you (using pyttsx3)

## Architecture

The demo is built with a modular backend architecture supporting two modes:

- **CPU Mode** (default): Uses CPU-based implementations
  - Speech-to-Text: `faster-whisper` with Whisper model
  - LLM: `transformers` with TinyLlama-1.1B-Chat
  - Text-to-Speech: `pyttsx3` (local TTS engine)

- **CPU-Rest TT-LLM Mode**: Uses Tenstorrent hardware acceleration for just the LLM phase
  - LLM: `tt-inference-server` with Llama-3.1-8B-Instruct

- **TT Mode** (planned): Uses Tenstorrent hardware acceleration (not yet implemented)

## Project Structure

```
quietbox-demo/
├── main.py                          # Main entry point and pipeline orchestration
├── requirements.txt                 # Python dependencies
└── backends/
    ├── wakeword_open.py            # Wake word detection (OpenWakeWord)
    ├── record_vad.py               # Voice Activity Detection recorder
    ├── stt_whisper_cpu.py          # Speech-to-text (Whisper CPU)
    ├── stt_tt.py                   # Speech-to-text (Tenstorrent - placeholder)
    ├── llm_hf_cpu.py               # LLM responder (TinyLlama CPU)
    ├── llm_tt.py                   # LLM responder (Tenstorrent - placeholder)
    └── tts_pyttsx3.py              # Text-to-speech (local TTS)
```

## How It Works

### Main Pipeline (`main.py`)

1. **Initialization**: Builds the pipeline components (wake word detector, recorder, STT, LLM, TTS)
2. **Main Loop**:
   - Waits for wake word detection ("hey quietbox" by default)
   - Records audio with VAD (stops after silence detection)
   - Transcribes audio to text using Whisper
   - Sends text to LLM for response generation
   - Speaks the response using TTS

### Backend Components

#### `wakeword_open.py` - Wake Word Detection
- Uses OpenWakeWord library for wake word detection
- Continuously monitors audio stream for wake word ("hey quietbox")
- Returns when wake word is detected with sufficient confidence

#### `record_vad.py` - Voice Activity Detection Recording
- Uses WebRTC VAD to detect speech activity
- Records audio frames until silence is detected
- Stops recording after 800ms of silence or 15 seconds maximum
- Saves audio as WAV file

#### `stt_whisper_cpu.py` - Speech-to-Text
- Uses `faster-whisper` library with Whisper model
- Runs on CPU with int8 quantization for performance
- Transcribes WAV audio file to text

#### `llm_hf_cpu.py` - LLM Response Generation
- Uses Hugging Face Transformers with TinyLlama-1.1B-Chat model
- Formats input with system/user/assistant prompt structure
- Generates responses with temperature sampling
- Extracts assistant response from generated text

#### `tts_pyttsx3.py` - Text-to-Speech
- Uses pyttsx3 for local text-to-speech synthesis
- No internet connection required
- Uses system TTS engine

## Installation

### Prerequisites

1. **Python 3.8+** (Python 3.10 tested)

2. **System Dependencies** (requires sudo):
   ```bash
   sudo apt-get update
   sudo apt-get install -y libportaudio2 portaudio19-dev
   ```

   Note: PortAudio is required for `sounddevice` to work (used for audio I/O).

### Python Dependencies

Install Python packages:
```bash
pip3 install -r requirements.txt
```

### First Run

The first time you run the demo, it will download:
- Whisper model weights (from Hugging Face)
- TinyLlama model weights (~2.3GB)
- OpenWakeWord models

These will be cached locally for subsequent runs.

## Usage

Run the demo:
```bash
python3 main.py
```

If using the TT backend, run:
```bash
export JWT_SECRET="testing" python3 main.py
```

The demo will:
1. Initialize all models (may take a minute on first run)
2. Print "QuietBox ready. Say your wake word to start."
3. Wait for you to say "hey quietbox" (or another wake word)
4. Listen for your question after detecting wake word
5. Process and respond to your question
6. Loop back to waiting for the next wake word

### Interactive Flow

1. **Wake Word**: Say "hey quietbox" to activate
2. **Speak**: After seeing "Listening for your question...", ask your question
3. **Response**: Wait for transcription, LLM processing, and audio response
4. **Repeat**: The system returns to waiting for the next wake word

### Example Session

```
QuietBox ready. Say your wake word to start.
Wake word detected. Listening for your question...
Transcribing...
User: What is Python?
Thinking...
Assistant: Python is a high-level programming language...
[Audio response plays]
Ready for the next wake word.
```

## Configuration

### Adjusting Model Parameters

In `main.py`, you can modify:
- Whisper model size: Change `model_size="small"` in `STTWhisperCPU` (options: tiny, base, small, medium, large)
- LLM model: Change `model_name` in `ResponderHFCPU` (must be compatible with ChatML format)
- Wake word threshold: Adjust `threshold=0.6` in `WakeWordDetector`
- Recording timeout: Change `max_seconds=15` in `Recorder`
- TTS speech rate: Adjust `rate=180` in `TTSLocal`

### Backend Selection

Currently only CPU mode is functional. To switch modes:
```python
run_loop(mode="cpu")  # or "tt" (when implemented)
```

## Troubleshooting

### Audio Issues

- **"PortAudio library not found"**: Install system dependencies (see Prerequisites)
- **No audio input**: Check microphone permissions and default audio device
- **Poor recognition**: Ensure quiet environment, speak clearly, check microphone quality

### Model Download Issues

- First run requires internet connection to download models
- Models are cached in `~/.cache/` directories
- Ensure sufficient disk space (~3-5GB for models)

### Performance

- Initial model loading takes time (30-60 seconds)
- CPU inference may be slow; consider using smaller models for faster responses
- TinyLlama responses are limited to 128 tokens by default

## Limitations

- **CPU-only inference**: Can be slow on CPU hardware
- **Local models**: Limited to models that fit in RAM
- **Simple VAD**: Basic voice activity detection; may miss quiet speech
- **No context memory**: Each interaction is independent (no conversation history)
- **Basic error handling**: May need manual restart if errors occur

## Future Enhancements

- Tenstorrent hardware acceleration (TT mode)
- Conversation context/memory
- Better error handling and recovery
- Configuration file support
- Multiple wake word options
- Streaming STT for faster responses

## License

This is a demo project. Check individual library licenses:
- OpenWakeWord: Apache 2.0
- Whisper: MIT
- TinyLlama: Apache 2.0
- transformers: Apache 2.0
- pyttsx3: Modified BSD License

