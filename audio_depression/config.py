import os

# Inference-only config.
# Training/data-split paths are intentionally NOT kept here.

# ===== PROJECT ROOT =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ===== INTERNAL DATA (Optional) =====
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# ===== MODEL ARTIFACTS (Place Colab-exported files here) =====
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_audio_model.pkl")
SCALER_INITIAL_PATH = os.path.join(MODELS_DIR, "scaler_initial.pkl")
SCALER_FINAL_PATH = os.path.join(MODELS_DIR, "scaler_final.pkl")
SELECTOR_PATH = os.path.join(MODELS_DIR, "feature_selector.pkl")

# Backward-compat alias (older code used feature_scaler.pkl)
SCALER_PATH = SCALER_FINAL_PATH
LEGACY_SCALER_FINAL_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")

# ===== AUDIO PARAMETERS =====
SAMPLE_RATE = 16000
# Max audio duration analyzed at inference time.
# Keeping this bounded prevents large STFT allocations for long uploads.
MAX_DURATION = 180  # seconds

# Feature extraction parameters (tuned to reduce memory pressure)
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
