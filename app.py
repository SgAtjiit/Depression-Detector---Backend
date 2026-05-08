from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import numpy as np

# Import prediction functions
from audio_depression.predict import predict_depression as predict_audio
from text_depression.text_predict import predict_depression_from_text

# ==================== FastAPI Setup ====================
app = FastAPI(
    title="Depression Detection API",
    description="Multi-modal API for detecting depression from audio and text inputs",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Response Models ====================
class TextRequest(BaseModel):
    text: str

class AudioResponse(BaseModel):
    prediction: int
    label: str
    probability: dict
    confidence: float
    message: str

class TextResponse(BaseModel):
    prediction: int
    label: str
    probability: dict
    confidence: float
    linguistic_features: dict
    message: str

class FusionResponse(BaseModel):
    prediction: int
    label: str
    ensemble_probability: dict
    confidence: float
    fusion_method: str
    individual_predictions: dict
    message: str

class HealthResponse(BaseModel):
    status: str
    audio_model: bool
    text_model: bool

# ==================== Helper Functions ====================
def get_risk_message(label: str, confidence: float) -> str:
    """Generate appropriate message based on prediction"""
    if label == "Depressed":
        if confidence >= 0.80:
            return " HIGH RISK: Strong indicators of depression detected. Please consider consulting a mental health professional immediately."
        elif confidence >= 0.60:
            return " MODERATE RISK: Signs of depression detected. We recommend speaking with a counselor or therapist."
        else:
            return " LOW RISK: Some concerning patterns detected. Consider monitoring your mental health."
    else:
        if confidence >= 0.80:
            return " LOW RISK: No significant signs of depression detected. Continue taking care of your mental health."
        else:
            return " INCONCLUSIVE: Results are mixed. Consider professional evaluation if you have concerns."

def fusion_predictions(audio_result: dict, text_result: dict, method: str = "weighted_average"):
    """
    Fuse audio and text predictions using different methods
    
    Methods:
    - weighted_average: 60% text, 40% audio (text is usually more reliable)
    - simple_average: 50% text, 50% audio
    - max_confidence: Use prediction with higher confidence
    - voting: Majority vote
    """
    audio_prob_depressed = audio_result['probability']['depressed']
    text_prob_depressed = text_result['probability']['depressed']
    
    if method == "weighted_average":
        # Weight: 60% text, 40% audio
        ensemble_prob = 0.6 * text_prob_depressed + 0.4 * audio_prob_depressed
            
    else:
        # Default to weighted average
        ensemble_prob = 0.6 * text_prob_depressed + 0.4 * audio_prob_depressed
    
    ensemble_prediction = 1 if ensemble_prob >= 0.5 else 0
    ensemble_label = "Depressed" if ensemble_prediction == 1 else "Not Depressed"
    
    return {
        "prediction": ensemble_prediction,
        "label": ensemble_label,
        "ensemble_probability": {
            "not_depressed": float(1 - ensemble_prob),
            "depressed": float(ensemble_prob)
        },
        "confidence": float(max(ensemble_prob, 1 - ensemble_prob)),
        "fusion_method": method
    }

# ==================== API Routes ====================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Check API health and model availability"""
    try:
        audio_model_exists = os.path.exists("audio_depression/models/best_audio_model.pkl")
        text_model_exists = os.path.exists("text_depression/models/best_text_model.pkl")
        
        return {
            "status": "healthy",
            "audio_model": audio_model_exists,
            "text_model": text_model_exists
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict/audio", response_model=AudioResponse)
async def predict_audio_endpoint(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG)")
):
    """
     AUDIO-ONLY PREDICTION
    
    Analyze audio file for depression indicators based on acoustic features:
    - Pitch variation
    - Speech rate
    - Energy levels
    - Spectral features (MFCC, mel-spectrogram)
    - Prosodic patterns
    
    **Input:** Audio file (WAV, MP3, FLAC, OGG, WEBM)
    
    **Output:** Depression prediction with confidence score
    """
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.webm'}
    file_ext = os.path.splitext(audio.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    temp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Predict using audio model
        result = predict_audio(temp_path)
        
        # Add risk message
        result['message'] = get_risk_message(result['label'], result['confidence'])
        
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Audio model not found. Please train the model first: {str(e)}"
        )
    except MemoryError as e:
        raise HTTPException(
            status_code=422,
            detail=(
                "Audio analysis failed due to insufficient memory while extracting features. "
                "Try a shorter clip (e.g., <= 60s) or a lower sample-rate file. "
                f"({type(e).__name__}: {e})"
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/predict/text", response_model=TextResponse)
async def predict_text_endpoint(request: TextRequest):
    """
     TEXT-ONLY PREDICTION
    
    Analyze text for depression indicators based on linguistic features:
    - Sentiment analysis
    - Negative word frequency
    - Syntactic patterns
    - TF-IDF features
    - Word usage patterns
    
    **Input:** Text string (conversation transcript, message, journal entry)
    
    **Output:** Depression prediction with confidence score and linguistic features
    """
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(
            status_code=400, 
            detail="Text is too short. Please provide at least 10 characters."
        )
    
    try:
        # Predict using text model
        result = predict_depression_from_text(request.text)
        
        # Add risk message
        result['message'] = get_risk_message(result['label'], result['confidence'])
        
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Text model not found. Please train the model first: {str(e)}"
        )
    except (ValueError, RuntimeError) as e:
        # ValueError: usually user input too short/invalid after cleaning
        # RuntimeError: usually server-side artifact mismatch
        status = 422 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status, detail=f"Text analysis failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.post("/predict/fusion", response_model=FusionResponse)
async def predict_fusion_endpoint(
    audio: UploadFile = File(..., description="Audio file"),
    text: str = Form(..., description="Text transcript of the audio"),
    fusion_method: str = Form("weighted_average", description="Fusion method: weighted_average, simple_average, max_confidence, voting")
):
    """
     MULTI-MODAL FUSION PREDICTION (MOST ACCURATE)
    
    Analyze BOTH audio and text together for maximum accuracy.
    Combines acoustic and linguistic features using ensemble methods.
    
    **Fusion Methods:**
    - `weighted_average` (default): 60% text + 40% audio (recommended)
    
    **Input:** 
    - Audio file (WAV, MP3, FLAC, OGG, WEBM)
    - Text transcript
    
    **Output:** Fused prediction with individual model results
    """
    # Validate inputs
    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text is too short (min 10 characters)")
    
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.webm'}
    file_ext = os.path.splitext(audio.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid audio file type. Supported: {', '.join(allowed_extensions)}"
        )
    
    if fusion_method not in ["weighted_average"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid fusion method. Choose: weighted_average, simple_average, max_confidence, voting"
        )
    
    temp_path = None
    try:
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Get predictions from both models
        audio_result = predict_audio(temp_path)
        text_result = predict_depression_from_text(text)
        
        # Fuse predictions
        fusion_result = fusion_predictions(audio_result, text_result, method=fusion_method)
        
        # Combine results
        combined_result = {
            **fusion_result,
            "individual_predictions": {
                "audio": {
                    "prediction": audio_result['prediction'],
                    "label": audio_result['label'],
                    "confidence": audio_result['confidence'],
                    "probability": audio_result['probability']
                },
                "text": {
                    "prediction": text_result['prediction'],
                    "label": text_result['label'],
                    "confidence": text_result['confidence'],
                    "probability": text_result['probability'],
                    "linguistic_features": text_result['linguistic_features']
                }
            },
            "message": get_risk_message(fusion_result['label'], fusion_result['confidence'])
        }
        
        return combined_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ==================== Additional Endpoints ====================

@app.get("/models/status")
async def models_status():
    """Check which models are available and their file sizes"""
    audio_model_path = "audio_depression/models/best_audio_model.pkl"
    text_model_path = "text_depression/models/best_text_model.pkl"
    
    def get_file_size(path):
        if os.path.exists(path):
            size_bytes = os.path.getsize(path)
            size_mb = size_bytes / (1024 * 1024)
            return f"{size_mb:.2f} MB"
        return "N/A"
    
    return {
        "audio_model": {
            "available": os.path.exists(audio_model_path),
            "path": audio_model_path,
            "size": get_file_size(audio_model_path)
        },
        "text_model": {
            "available": os.path.exists(text_model_path),
            "path": text_model_path,
            "size": get_file_size(text_model_path)
        }
    }

@app.get("/info")
async def api_info():
    """Get API information and usage instructions"""
    return {
        "name": "Depression Detection API",
        "version": "2.0.0",
        "description": "Multi-modal depression detection using audio and text analysis",
        "endpoints": {
            "/": "GET - Health check",
            "/predict/audio": "POST - Audio-only prediction",
            "/predict/text": "POST - Text-only prediction",
            "/predict/fusion": "POST - Multi-modal fusion prediction (RECOMMENDED)",
            "/models/status": "GET - Check model availability",
            "/info": "GET - API information",
            "/docs": "GET - Interactive API documentation"
        },
        "supported_audio_formats": [".wav", ".mp3", ".flac", ".ogg", ".webm"],
        "fusion_methods": [
            "weighted_average (60% text, 40% audio) - RECOMMENDED",
        ],
        "note": "Visit /docs for interactive API testing with Swagger UI"
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" Starting Depression Detection API Server")
    print("="*60)
    print(" API will be available at: http://localhost:8000")
    print(" Interactive docs at: http://localhost:8000/docs")
    print(" Alternative docs at: http://localhost:8000/redoc")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)