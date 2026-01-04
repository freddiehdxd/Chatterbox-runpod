"""
Chatterbox TTS - RunPod Serverless Handler
https://github.com/resemble-ai/chatterbox

Supports:
- Text-to-speech generation
- Voice cloning via audio prompt
- CFG weight and exaggeration tuning
- R2/S3 upload or base64 output
"""

import runpod
import base64
import uuid
import os
import logging
import requests
import tempfile
import torch
import torchaudio as ta
import soundfile as sf
import boto3
from io import BytesIO

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# R2/S3 Configuration (Optional)
# --------------------------------------------------
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET", "cdn")
CDN_URL = os.getenv("CDN_URL", "https://cdn.example.com")

# Initialize R2/S3 client
r2_client = None
if R2_ENDPOINT and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    try:
        r2_client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto'
        )
        logger.info("✅ R2/S3 client initialized")
    except Exception as e:
        logger.error(f"❌ R2/S3 initialization failed: {e}")
else:
    logger.info("ℹ️ R2/S3 not configured - will return base64 audio")

# --------------------------------------------------
# Paths
# --------------------------------------------------
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# GPU Check (GPU Only - no CPU fallback)
# --------------------------------------------------
def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is required but not available. This handler requires a GPU.")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
    return "cuda"

DEVICE = check_gpu()

# --------------------------------------------------
# Load Chatterbox Model (once at startup)
# --------------------------------------------------
logger.info("🔄 Loading Chatterbox TTS model...")
try:
    from chatterbox.tts import ChatterboxTTS
    MODEL = ChatterboxTTS.from_pretrained(device=DEVICE)
    SAMPLE_RATE = MODEL.sr
    logger.info(f"✅ Chatterbox TTS loaded (sample rate: {SAMPLE_RATE}Hz)")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    MODEL = None
    SAMPLE_RATE = 24000

# --------------------------------------------------
# Upload to R2/S3
# --------------------------------------------------
def upload_to_r2(file_path, job_id, file_ext="wav"):
    """Upload audio file to R2/S3 and return public URL"""
    if not r2_client:
        return None
    
    try:
        file_name = f"tts/{job_id}.{file_ext}"
        content_type = "audio/wav" if file_ext == "wav" else "audio/mpeg"
        
        with open(file_path, 'rb') as f:
            r2_client.put_object(
                Bucket=R2_BUCKET,
                Key=file_name,
                Body=f,
                ContentType=content_type
            )
        
        public_url = f"{CDN_URL}/{file_name}"
        logger.info(f"✅ Uploaded to R2: {public_url}")
        return public_url
    
    except Exception as e:
        logger.error(f"❌ R2 upload failed: {e}")
        return None

# --------------------------------------------------
# Download audio from URL
# --------------------------------------------------
def download_audio(url, save_path):
    """Download audio file from URL"""
    if not url.startswith(("http://", "https://")):
        raise ValueError("Invalid URL")
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        f.write(response.content)
    
    return save_path

# --------------------------------------------------
# Handler
# --------------------------------------------------
def handler(job):
    """
    RunPod Serverless Handler for Chatterbox TTS
    
    Input parameters:
    - text (str, required): Text to synthesize
    - audio_prompt (str, optional): URL or base64 encoded audio for voice cloning (auto-detected)
    - exaggeration (float, optional): Expressiveness control, default 0.5 (0.0-1.0+)
    - cfg_weight (float, optional): Classifier-free guidance weight, default 0.5 (0.0-1.0)
    - output_format (str, optional): "wav" or "mp3", default "wav"
    - return_base64 (bool, optional): Force base64 return even if R2 configured
    
    Tips from Chatterbox docs:
    - Default settings (exaggeration=0.5, cfg_weight=0.5) work well for most prompts
    - For expressive/dramatic speech: lower cfg_weight (~0.3), higher exaggeration (~0.7+)
    - If reference speaker is fast, lower cfg_weight to ~0.3 for better pacing
    """
    job_input = job.get("input", {})
    job_id = job.get("id", str(uuid.uuid4()))
    
    logger.info(f"📥 Job received: {job_id}")
    
    # Validate model is loaded
    if MODEL is None:
        return {"error": "Model failed to load at startup"}
    
    # --------------------------------------------------
    # Extract parameters
    # --------------------------------------------------
    text = job_input.get("text", "").strip()
    if not text:
        return {"error": "Missing required parameter: text"}
    
    # Generation parameters
    exaggeration = float(job_input.get("exaggeration", 0.5))
    cfg_weight = float(job_input.get("cfg_weight", 0.5))
    output_format = job_input.get("output_format", "wav").lower()
    return_base64 = job_input.get("return_base64", False)
    
    if output_format not in ["wav", "mp3"]:
        output_format = "wav"
    
    logger.info(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    logger.info(f"⚙️ Settings: exaggeration={exaggeration}, cfg_weight={cfg_weight}")
    
    # --------------------------------------------------
    # Handle audio prompt for voice cloning
    # Auto-detects URL vs base64
    # --------------------------------------------------
    audio_prompt_path = None
    temp_audio_file = None
    
    try:
        if "audio_prompt" in job_input:
            audio_prompt_value = job_input["audio_prompt"].strip()
            
            # Auto-detect: URL or base64
            if audio_prompt_value.startswith(("http://", "https://")):
                # It's a URL - download it
                try:
                    temp_audio_file = os.path.join(INPUT_DIR, f"{job_id}_prompt.wav")
                    download_audio(audio_prompt_value, temp_audio_file)
                    audio_prompt_path = temp_audio_file
                    logger.info(f"🎤 Audio prompt downloaded from URL")
                except Exception as e:
                    return {"error": f"Failed to download audio prompt: {str(e)}"}
            else:
                # Assume base64
                try:
                    audio_bytes = base64.b64decode(audio_prompt_value)
                    temp_audio_file = os.path.join(INPUT_DIR, f"{job_id}_prompt.wav")
                    with open(temp_audio_file, 'wb') as f:
                        f.write(audio_bytes)
                    audio_prompt_path = temp_audio_file
                    logger.info(f"🎤 Audio prompt loaded from base64")
                except Exception as e:
                    return {"error": f"Invalid base64 audio_prompt: {str(e)}"}
        
        # --------------------------------------------------
        # Generate speech
        # --------------------------------------------------
        logger.info("🔊 Generating speech...")
        
        try:
            if audio_prompt_path:
                wav = MODEL.generate(
                    text,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            else:
                wav = MODEL.generate(
                    text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight
                )
            
            logger.info(f"✅ Audio generated: shape={wav.shape}, duration={wav.shape[-1]/SAMPLE_RATE:.2f}s")
        
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return {"error": f"TTS generation failed: {str(e)}"}
        
        # --------------------------------------------------
        # Save output audio
        # --------------------------------------------------
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.{output_format}")
        
        try:
            if output_format == "wav":
                ta.save(output_path, wav, SAMPLE_RATE)
            else:
                # Save as WAV first, then convert to MP3
                temp_wav = os.path.join(OUTPUT_DIR, f"{job_id}_temp.wav")
                ta.save(temp_wav, wav, SAMPLE_RATE)
                
                # Convert to MP3 using ffmpeg
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_wav, 
                    "-codec:a", "libmp3lame", "-qscale:a", "2",
                    output_path
                ], check=True, capture_output=True)
                os.remove(temp_wav)
            
            logger.info(f"💾 Audio saved: {output_path}")
        
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            return {"error": f"Failed to save audio: {str(e)}"}
        
        # --------------------------------------------------
        # Return result (R2 upload or base64)
        # --------------------------------------------------
        result = {
            "duration": float(wav.shape[-1] / SAMPLE_RATE),
            "sample_rate": SAMPLE_RATE,
            "format": output_format
        }
        
        if r2_client and not return_base64:
            public_url = upload_to_r2(output_path, job_id, output_format)
            if public_url:
                result["audio_url"] = public_url
                # Clean up local file
                try:
                    os.remove(output_path)
                except:
                    pass
                return result
        
        # Fallback to base64
        with open(output_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        result["audio"] = audio_b64
        
        # Clean up
        try:
            os.remove(output_path)
        except:
            pass
        
        return result
    
    finally:
        # Clean up temp audio prompt file
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
            except:
                pass

# --------------------------------------------------
# RunPod Serverless start
# --------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Chatterbox TTS Serverless Handler")
    runpod.serverless.start({"handler": handler})
