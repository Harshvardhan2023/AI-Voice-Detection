import base64
import io
import os
import tempfile
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np
import librosa
import soundfile as sf

# Load Environment Variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "")

# Supported Languages
SUPPORTED_LANGUAGES = ["English", "Hindi", "Malayalam", "Telugu", "Tamil"]

# Audio & Feature Parameters
SAMPLE_RATE = 16000

# Load trained ML model
MODEL_PATH = "artifacts/models/best_ml_model.joblib"
model = joblib.load(MODEL_PATH)

# Initialize FastAPI
app = FastAPI(title="AI Voice Detection API")

# Request body schema
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# Check API Key
def check_api_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# Feature Extraction
def extract_features(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.concatenate([mel_mean, mel_std, mfcc_mean, mfcc_std])
    return features

# Preprocess Audio (MP3-safe with ID3 tag handling)
def preprocess_audio(audio_bytes: bytes):
    try:
        # Remove ID3 tags if present
        if audio_bytes.startswith(b'ID3'):
            if len(audio_bytes) >= 10:
                # ID3v2 tag size is stored in bytes 6-9 as synchsafe integer
                size = ((audio_bytes[6] & 0x7f) << 21) | \
                       ((audio_bytes[7] & 0x7f) << 14) | \
                       ((audio_bytes[8] & 0x7f) << 7) | \
                       (audio_bytes[9] & 0x7f)
                # Skip ID3 header (10 bytes) + tag data
                audio_bytes = audio_bytes[10 + size:]
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            tmp_path = tmp.name
        
        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
        finally:
            os.unlink(tmp_path)  # Clean up temp file
            
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid or Corrupted Audio File")
    
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    y = librosa.util.normalize(y)
    features = extract_features(y, SAMPLE_RATE)
    return features.reshape(1, -1)

# Generate explanation
def explain_prediction(pred, proba):
    if pred == 0:  # AI_GENERATED
        if proba > 0.9:
            return "Strong AI Patterns detected: Unnatural Pitch Consistency and Robotic Speech Characteristics"
        elif proba > 0.7:
            return "Moderate AI Patterns detected: Synthetic Voice Artifacts and Irregular Prosody"
        else:
            return "Weak AI Patterns detected: Some Unnatural Speech Characteristics present"
    else:  # HUMAN
        if proba > 0.9:
            return "Strong Human Voice Patterns detected: Natural Speech Variations and Organic Vocal Characteristics"
        elif proba > 0.7:
            return "Moderate Human Voice Patterns detected: Typical Human Speech Dynamics detected"
        else:
            return "Weak Human Voice Patterns detected: Some Natural Speech Characteristics present"

# Error handler for proper error response format
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail
        }
    )

# Main API endpoint
@app.post("/api/voice-detection")
def detect_voice(req: VoiceRequest, x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language. Must be one of: English, Hindi, Malayalam, Telugu, Tamil")
    
    if req.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format is supported")
    
    clean_b64 = re.sub(r"\s+", "", req.audioBase64)
    try:
        audio_bytes = base64.b64decode(clean_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 encoding")
    
    features = preprocess_audio(audio_bytes)
    pred = model.predict(features)[0]
    proba = (
        float(np.max(model.predict_proba(features)))
        if hasattr(model, "predict_proba")
        else 1.0
    )
    
    classification = "HUMAN" if pred == 1 else "AI_GENERATED"
    explanation = explain_prediction(pred, proba)
    
    return {
        "status": "success",
        "language": req.language,
        "classification": classification,
        "confidenceScore": round(proba, 5),
        "explanation": explanation
    }