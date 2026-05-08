import os
import tempfile
from io import BytesIO
from functools import lru_cache

import joblib
import librosa
import pandas as pd
import soundfile as sf

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None


class AudioConversionError(RuntimeError):
    pass

try:
    # Package import
    from .config import (
        BEST_MODEL_PATH,
        SCALER_INITIAL_PATH,
        SCALER_FINAL_PATH,
        SELECTOR_PATH,
        LEGACY_SCALER_FINAL_PATH,
        SAMPLE_RATE,
        MAX_DURATION,
        N_FFT,
        HOP_LENGTH,
        N_MELS,
    )
    from .feature_extraction import extract_features_from_file
except Exception:
    # Script import fallback
    from config import (
        BEST_MODEL_PATH,
        SCALER_INITIAL_PATH,
        SCALER_FINAL_PATH,
        SELECTOR_PATH,
        LEGACY_SCALER_FINAL_PATH,
        SAMPLE_RATE,
        MAX_DURATION,
        N_FFT,
        HOP_LENGTH,
        N_MELS,
    )
    from feature_extraction import extract_features_from_file

def convert_audio_to_wav(audio_path_or_bytes, output_path: str) -> str:
    """
    Convert any audio format to WAV (handles webm, mp3, etc.)
    Works with both file paths and byte streams
    """
    if AudioSegment is None:
        raise AudioConversionError(
            "Audio conversion requires ffmpeg. Either upload a WAV file, or install ffmpeg and ensure it's on PATH."
        )

    try:
        if isinstance(audio_path_or_bytes, (bytes, BytesIO)):
            stream = BytesIO(audio_path_or_bytes) if isinstance(audio_path_or_bytes, bytes) else audio_path_or_bytes
            audio = AudioSegment.from_file(stream)
        else:
            audio = AudioSegment.from_file(audio_path_or_bytes)

        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        raise AudioConversionError(
            f"Failed to decode/convert audio to WAV. Install ffmpeg and retry, or upload WAV. ({type(e).__name__}: {e})"
        ) from e

def _resolve_scaler_final_path() -> str:
    if os.path.exists(SCALER_FINAL_PATH):
        return SCALER_FINAL_PATH
    if os.path.exists(LEGACY_SCALER_FINAL_PATH):
        return LEGACY_SCALER_FINAL_PATH
    return SCALER_FINAL_PATH


@lru_cache(maxsize=1)
def load_model_and_artifacts():
    """Load trained model, scalers, and feature selector (cached)."""
    scaler_final_path = _resolve_scaler_final_path()
    required_files = {
        "model": BEST_MODEL_PATH,
        "scaler_initial": SCALER_INITIAL_PATH,
        "scaler_final": scaler_final_path,
        "selector": SELECTOR_PATH,
    }

    missing = [f"{name}: {path}" for name, path in required_files.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifacts in audio_depression/models.\n"
            "Place these files (exported from Colab training) and retry:\n- "
            + "\n- ".join(missing)
        )

    model = joblib.load(BEST_MODEL_PATH)
    scaler_initial = joblib.load(SCALER_INITIAL_PATH)
    scaler_final = joblib.load(scaler_final_path)
    selector = joblib.load(SELECTOR_PATH)
    return model, scaler_initial, scaler_final, selector

def predict_depression(audio_path_or_bytes):
    """
    Predict depression from audio file or bytes
     Automatically handles variable duration audio
     Converts any format to WAV
    Returns: dict with prediction, probabilities, and confidence
    """
    temp_file = None
    
    try:
        # Convert audio to WAV if needed
        if isinstance(audio_path_or_bytes, (bytes, BytesIO)):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            temp_file = tmp.name
            audio_path = convert_audio_to_wav(audio_path_or_bytes, temp_file)
        else:
            if not os.path.exists(audio_path_or_bytes):
                raise FileNotFoundError(f"Audio file not found: {audio_path_or_bytes}")

            if not str(audio_path_or_bytes).lower().endswith(".wav"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp.close()
                temp_file = tmp.name
                audio_path = convert_audio_to_wav(audio_path_or_bytes, temp_file)
            else:
                audio_path = audio_path_or_bytes

        if not str(audio_path).lower().endswith(".wav"):
            raise AudioConversionError(
                "Audio must be WAV for feature extraction. Upload WAV or enable conversion by installing ffmpeg."
            )
        
        model, scaler_initial, scaler_final, selector = load_model_and_artifacts()
        
        # Get audio duration using soundfile (more reliable)
        try:
            audio_info = sf.info(audio_path)
            audio_duration = audio_info.duration
        except Exception as e:
            audio_duration = librosa.get_duration(path=audio_path)
        
        # Extract features (bounded duration + lower-memory STFT params)
        features_dict = extract_features_from_file(audio_path,sr_target=16000,max_duration=MAX_DURATION,)
        
        # Remove audio_duration if present (not a feature for model)
        actual_duration = features_dict.pop('audio_duration', audio_duration)
        
        features_df = pd.DataFrame([features_dict])
        
        # Apply pipeline
        features_scaled = scaler_initial.transform(features_df)
        features_selected = selector.transform(features_scaled)
        features_final = scaler_final.transform(features_selected)
        
        # Predict
        prediction = model.predict(features_final)[0]
        probability = model.predict_proba(features_final)[0]
        
        result = {
            'prediction': int(prediction),
            'label': 'Depressed' if prediction == 1 else 'Not Depressed',
            'probability': {
                'not_depressed': float(probability[0]),
                'depressed': float(probability[1])
            },
            'confidence': float(max(probability)),
            'audio_duration': float(audio_duration),
            'analyzed_duration': float(actual_duration)
        }
        return result
    
    except Exception:
        raise
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        try:
            result = predict_depression(audio_path)

            print("\n" + "=" * 60)
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"P(Not Depressed): {result['probability']['not_depressed']:.2%}")
            print(f"P(Depressed):     {result['probability']['depressed']:.2%}")
            print("=" * 60)

        except Exception as e:
            print(f"\nError: {str(e)}")
            sys.exit(1)
    else:
        print("Usage: python predict.py <path_to_audio_file>")
        print("\nExample:")
        print('  python predict.py "D:/depressiondetector/test_audio.wav"')