# AI Voice Detection (ML-based)

**Detect whether a voice sample is AI-generated or spoken by a human across 5 Indian languages.**  

Supported Languages: **Tamil, English, Hindi, Malayalam, Telugu**  

This repository contains the code, trained model, and API implementation for real-time detection of AI-generated voices.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Modeling](#modeling)  
- [API](#api)  
- [Installation](#installation)  
- [Usage](#usage)  

---

## Overview

Modern AI systems can generate highly realistic voices. This project implements a **Machine Learning pipeline** that classifies audio clips as either **AI-generated** or **Human**.  

The pipeline includes:

- **Feature extraction** using MFCC  
- **Classical ML models**: Logistic Regression, Random Forest, SVM  
- **Trained model** saved for inference  
- **FastAPI-based REST API** for online detection  

---

## Features

- Accepts **Base64-encoded MP3 audio**  
- Supports 5 Indian languages  
- Returns **classification, confidence score, and explanation**  
- Secure API access using **API Key**  
- Handles audio of varying lengths  

---

## Dataset

- **Human voices**: Indian Languages Audio Dataset (~1,000 samples per language) + English samples  
- **AI-generated voices**: Synthesized using TTS systems  
- Each audio clip: **5 seconds**, **16 kHz**  

Directory Structure:
```kotlin
data/
├─ AI/
|  ├─ English/
│  ├─ Hindi/
|  ├─ Malayalam/
|  ├─ Telugu/
│  └─ Tamil/
├─ Human/
|  ├─ English/
│  ├─ Hindi/
|  ├─ Malayalam/
|  ├─ Telugu/
│  └─ Tamil/
```

---

## Modeling

- **Notebook 1:** Audio preprocessing, waveform visualization, Mel Spectrogram, MFCC extraction  
- **Notebook 2:** Feature extraction (MFCC) and saving `.npy` feature arrays  
- **Notebook 3:** ML models training (Logistic Regression, Random Forest, SVM), evaluation, and best model selection  

---

## API

- Implemented using **FastAPI**  
- Endpoint: `POST /api/voice-detection`  
- Request Body:

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "<Base64 MP3>"
}
```

- Response:
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

- Secured with x-api-key header

## Installation
```bash
git clone <repo_url>
cd <repo_name>
python -m venv venv
# Activate virtual environment
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the API locally:
```bash
uvicorn main:app --reload
```

Send a request (example with cURL):
```bash
curl -X POST http://127.0.0.1:8000/api/voice-detection \
-H "Content-Type: application/json" \
-H "x-api-key: <YOUR_API_KEY>" \
-d '{"language": "Hindi", "audioFormat": "mp3", "audioBase64": "<BASE64_AUDIO>"}'
```
