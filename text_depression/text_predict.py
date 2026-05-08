import os
from functools import lru_cache

import joblib
import numpy as np

try:
    # Package / API usage (imported as text_depression.text_predict)
    from .text_config import BEST_MODEL_PATH, MODELS_DIR, SCALER_PATH, VECTORIZER_PATH
except Exception:
    # Script usage fallback (python text_predict.py ...)
    from text_config import BEST_MODEL_PATH, MODELS_DIR, SCALER_PATH, VECTORIZER_PATH


def clean_text(text: str) -> str:
    """Match the cleaning approach used in the notebook.

    - Remove timestamps like "2.1,3.2," and standalone floats
    - Remove confidence fragments like ", 0.9876"
    - Remove standalone integers
    - Keep basic punctuation; lowercase
    """
    if text is None:
        return ""
    text = str(text)
    if text.strip() == "":
        return ""

    import re

    text = re.sub(r"\d+\.\d+,\s*\d+\.\d+,", "", text)
    text = re.sub(r",\s*0\.\d+", "", text)
    text = re.sub(r"\d+\.\d+,", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r",+", ",", text)
    text = text.strip(",")
    text = re.sub(r"[^a-zA-Z\s\.\,\?\!\'\-]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@lru_cache(maxsize=1)
def load_text_model_and_artifacts():
    """Load trained text model artifacts (cached)."""
    model_path = BEST_MODEL_PATH
    vectorizer_path = VECTORIZER_PATH
    scaler_path = SCALER_PATH
    selector_path = os.path.join(MODELS_DIR, "feature_selector.pkl")
    
    required_files = {
        'model': model_path,
        'vectorizer': vectorizer_path,
        'scaler': scaler_path,
        'selector': selector_path
    }
    
    missing_files = []
    for name, path in required_files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(
            "Missing required model artifacts in text_depression/models.\n"
            "Place these files (exported from your training notebook) and retry:\n- "
            + "\n- ".join(missing_files)
        )
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)

    # Basic compatibility checks to avoid runtime 500s later.
    # These artifacts must come from the same training run.
    try:
        vec_dim = vectorizer.transform(["sanity check"]).shape[1]
    except Exception:
        vec_dim = getattr(vectorizer, "n_features_in_", None)

    sel_dim = getattr(selector, "n_features_in_", None)
    if vec_dim is not None and sel_dim is not None and int(vec_dim) != int(sel_dim):
        raise RuntimeError(
            "Text model artifacts mismatch: the vectorizer produces "
            f"{int(vec_dim)} features but the feature selector expects {int(sel_dim)}. "
            "Re-export `best_text_model.pkl`, `text_vectorizer.pkl`, `text_scaler.pkl`, and `feature_selector.pkl` "
            "from the same training run (do not mix files from different runs)."
        )

    # If scaler has expected input features, validate selector output size.
    scaler_dim = getattr(scaler, "n_features_in_", None)
    sel_k = getattr(selector, "k", None)
    if scaler_dim is not None and sel_k is not None and int(scaler_dim) != int(sel_k):
        raise RuntimeError(
            "Text model artifacts mismatch: the scaler expects "
            f"{int(scaler_dim)} features but the selector is configured for k={int(sel_k)}. "
            "Re-export artifacts from the same training run."
        )
    
    return model, vectorizer, scaler, selector

def extract_linguistic_features(text):
    """Extract simple linguistic features for interpretation"""
    words = text.split()
    sentences = text.split('.')
    
    # Depression-related keywords
    negative_words = ['sad', 'depressed', 'hopeless', 'alone', 'tired', 'empty', 'worthless', 'anxious', 'nervous', 'worry']
    positive_words = ['happy', 'good', 'great', 'excited', 'wonderful', 'love', 'enjoy', 'fun', 'glad', 'pleased']
    
    negative_count = sum(1 for word in words if word.lower() in negative_words)
    positive_count = sum(1 for word in words if word.lower() in positive_words)
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'negative_word_count': negative_count,
        'positive_word_count': positive_count,
        'sentiment_ratio': (positive_count - negative_count) / max(len(words), 1)
    }

def predict_depression_from_text(text):
    """
    Predict depression from text input
    Returns: dict with prediction, probabilities, confidence, and features
    """
    if not text or len(text.strip()) < 10:
        raise ValueError("Text is too short. Please provide at least 10 characters.")
    
    model, vectorizer, scaler, selector = load_text_model_and_artifacts()

    cleaned_text = clean_text(text)
    if len(cleaned_text) < 10:
        raise ValueError("Text is too short after cleaning. Please provide more content.")

    # Apply notebook-equivalent pipeline
    text_tfidf = vectorizer.transform([cleaned_text])
    text_selected = selector.transform(text_tfidf)
    text_scaled = scaler.transform(text_selected.toarray())
    
    # Predict
    prediction = model.predict(text_scaled)[0]
    probability = model.predict_proba(text_scaled)[0]
    
    # Extract linguistic features
    ling_features = extract_linguistic_features(cleaned_text)
    
    result = {
        'prediction': int(prediction),
        'label': 'Depressed' if prediction == 1 else 'Not Depressed',
        'probability': {
            'not_depressed': float(probability[0]),
            'depressed': float(probability[1])
        },
        'confidence': float(max(probability)),
        'linguistic_features': ling_features
    }
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        text_input = ' '.join(sys.argv[1:])
        
        try:
            result = predict_depression_from_text(text_input)
            
            print("\n" + "="*60)
            print(f"🎯 Prediction: {result['label']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print(f"\n📈 Probabilities:")
            print(f"   Not Depressed: {result['probability']['not_depressed']:.2%}")
            print(f"   Depressed:     {result['probability']['depressed']:.2%}")
            print(f"\n📝 Linguistic Features:")
            for key, val in result['linguistic_features'].items():
                print(f"   {key}: {val}")
            print("="*60)
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            sys.exit(1)
    else:
        print("Usage: python text_predict.py <text>")
        print("\nExample:")
        print('  python text_predict.py "I feel very sad and hopeless today"')