# Chatterbox TTS - RunPod Serverless

RunPod serverless handler for [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) - State-of-the-art open-source text-to-speech model from Resemble AI.

## Features

- ✅ Text-to-speech generation
- ✅ Voice cloning via audio prompt
- ✅ CFG weight & exaggeration tuning
- ✅ WAV/MP3 output formats
- ✅ Optional R2/S3 upload
- ✅ Base64 audio output fallback

## Model Info

| Model | Size | Languages | Key Features |
|-------|------|-----------|--------------|
| Chatterbox | 500M | English | CFG & Exaggeration tuning |

## Build & Deploy

### Build Docker Image

```bash
cd chatterbox
docker build -t chatterbox-tts:latest .
```

### Push to Docker Hub

```bash
docker tag chatterbox-tts:latest your-dockerhub/chatterbox-tts:latest
docker push your-dockerhub/chatterbox-tts:latest
```

### Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create new endpoint
3. Use your Docker image: `your-dockerhub/chatterbox-tts:latest`
4. Recommended GPU: RTX 4090 / A10G / L4 (8GB+ VRAM)
5. Set environment variables (optional, for R2 storage):
   - `R2_ENDPOINT`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET`
   - `CDN_URL`

## API Usage

### Basic TTS

```json
{
  "input": {
    "text": "Hello, this is a test of Chatterbox text-to-speech."
  }
}
```

### With Voice Cloning (URL)

```json
{
  "input": {
    "text": "Hello, this should sound like the reference voice.",
    "audio_prompt": "https://example.com/reference-voice.wav"
  }
}
```

### With Voice Cloning (Base64)

```json
{
  "input": {
    "text": "Hello, this should sound like the reference voice.",
    "audio_prompt": "<base64-encoded-audio>"
  }
}
```

### Full Parameters

```json
{
  "input": {
    "text": "Your text to synthesize",
    "audio_prompt": "https://example.com/voice.wav",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "output_format": "wav",
    "return_base64": false
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `audio_prompt` | string | - | URL or base64 audio for voice cloning (auto-detected) |
| `exaggeration` | float | 0.5 | Expressiveness control (0.0 - 1.0+) |
| `cfg_weight` | float | 0.5 | Classifier-free guidance (0.0 - 1.0) |
| `output_format` | string | "wav" | Output format: "wav" or "mp3" |
| `return_base64` | bool | false | Force base64 return even if R2 configured |

### Response

With R2 configured:
```json
{
  "audio_url": "https://cdn.example.com/tts/job-id.wav",
  "duration": 2.5,
  "sample_rate": 24000,
  "format": "wav"
}
```

Without R2 (base64):
```json
{
  "audio": "<base64-encoded-audio>",
  "duration": 2.5,
  "sample_rate": 24000,
  "format": "wav"
}
```

## Tips from Chatterbox Docs

- **Default settings** (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts
- **Expressive/Dramatic speech**: Lower `cfg_weight` (~0.3), higher `exaggeration` (~0.7+)
- **Fast reference speaker**: Lower `cfg_weight` to ~0.3 for better pacing
- Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate

## Local Testing

```bash
# Test locally with Docker
docker run -it --gpus all \
  -e RUNPOD_DEBUG=1 \
  chatterbox-tts:latest \
  python -u /app/handler.py

# Then send test request
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

## License

Chatterbox TTS is MIT licensed. See the [original repository](https://github.com/resemble-ai/chatterbox) for details.

## Watermarking

Chatterbox includes [PerTh watermarking](https://github.com/resemble-ai/chatterbox#built-in-perth-watermarking-for-responsible-ai) - all generated audio contains imperceptible neural watermarks for responsible AI use.
