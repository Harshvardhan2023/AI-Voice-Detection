import base64
import io
import os
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
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
        raise HTTPException(status_code=401, detail="Invalid API Key !!")

# Feature Extraction
def extract_features(y, sr):
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=sr // 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Combine features
    features = np.concatenate([
        mel_mean, mel_std,
        mfcc_mean, mfcc_std
    ])
    return features

# Preprocess audio (MP3-safe)
def preprocess_audio(audio_bytes: bytes):
    try:
        # Write MP3 bytes to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()

            # Decode MP3 using librosa (ffmpeg-backed)
            y, sr = librosa.load(tmp.name, sr=None, mono=True)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Audio File !!")

    # Resample
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Normalize amplitude
    y = librosa.util.normalize(y)

    # Extract features
    features = extract_features(y, SAMPLE_RATE)
    return features.reshape(1, -1)

# Generate explanation for prediction
def explain_prediction(pred, proba):
    if pred == 0:
        return f"AI-generated Patterns Detected (Confidence {proba:.2f})"
    else:
        return f"Human Voice Patterns Detected (Confidence {proba:.2f})"

# Main API endpoint
@app.post("/api/voice-detection")
def detect_voice(req: VoiceRequest, x_api_key: str = Header(...)):
    # Validate API key
    check_api_key(x_api_key)

    # Validate Language
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported Language")

    # Validate Audio Format
    if req.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 Audio supported !!")

    # Decode Base64 Audio (STRICT)
    try:
        audio_bytes = base64.b64decode(req.audioBase64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed Base64 Audio !!")

    # Preprocess & extract features
    features = preprocess_audio(audio_bytes)

    # Predict
    pred = model.predict(features)[0]
    proba = (
        float(np.max(model.predict_proba(features)))
        if hasattr(model, "predict_proba")
        else 1.0
    )

    # Map prediction to required output
    classification = "HUMAN" if pred == 1 else "AI_GENERATED"
    explanation = explain_prediction(pred, proba)

    return {
        "status": "success",
        "language": req.language,
        "classification": classification,
        "confidenceScore": proba,
        "explanation": explanation
    }