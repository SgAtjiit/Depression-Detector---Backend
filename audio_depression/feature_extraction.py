import numpy as np
import librosa
import soundfile as sf

try:
    # Package import
    from .config import SAMPLE_RATE, MAX_DURATION, N_FFT, HOP_LENGTH, N_MELS
except Exception:
    # Script import fallback
    from config import SAMPLE_RATE, MAX_DURATION, N_FFT, HOP_LENGTH, N_MELS

def aggregate_stats(arr):
    """Return mean, std, min, max for a numpy array (flattened)."""
    arr_flat = np.asarray(arr).flatten()
    if arr_flat.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (np.mean(arr_flat), np.std(arr_flat), np.min(arr_flat), np.max(arr_flat))

def extract_features_from_file(file_path, sr_target=16000, max_duration=180):
    """
    Features extracted:
    1. MFCCs (Mel-Frequency Cepstral Coefficients) - 20 coefficients
    2. Chroma features - Pitch class representation
    3. Mel spectrogram - Time-frequency representation
    4. Spectral contrast - Spectral peak vs valley
    5. Tonnetz - Tonal centroid features
    6. Pitch (F0) - Fundamental frequency
    7. Energy (RMS) - Vocal energy
    8. Zero Crossing Rate - Speech activity
    9. Spectral features - Centroid, rolloff, bandwidth
    10. Pause detection - Silence ratio and pauses
    11. Onset strength - Speech rate indicator
    """
    try:
        audio_info = sf.info(file_path)
        audio_duration = audio_info.duration
    except Exception as e:
        audio_duration = librosa.get_duration(path=file_path)

    # duration handling- if less than 180 s then taken otherwise trimmed to 180 s
    if audio_duration > max_duration:
        duration_to_load = max_duration
    else:
        duration_to_load = None
    # Load audio
    try:
        y, sr = librosa.load(file_path, sr=sr_target, duration=duration_to_load)
    except Exception as e:
        # Fallback method
        y, sr = sf.read(file_path, dtype='float32')
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        if duration_to_load:
            y = y[:int(duration_to_load * sr)]

    # Trim silence
    y, _ = librosa.effects.trim(y)
    feats = {}
    # MFCCs (20 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats['mfcc_mean'], feats['mfcc_std'], feats['mfcc_min'], feats['mfcc_max'] = aggregate_stats(mfccs)

    #Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats['chroma_mean'], feats['chroma_std'], feats['chroma_min'], feats['chroma_max'] = aggregate_stats(chroma)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    feats['mel_mean'], feats['mel_std'], feats['mel_min'], feats['mel_max'] = aggregate_stats(mel)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats['contrast_mean'], feats['contrast_std'], feats['contrast_min'], feats['contrast_max'] = aggregate_stats(contrast)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    feats['tonnetz_mean'], feats['tonnetz_std'], feats['tonnetz_min'], feats['tonnetz_max'] = aggregate_stats(tonnetz)

    # Pitch (F0) - Critical for depression detection
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 0:
            feats['pitch_mean'] = np.mean(f0_valid)
            feats['pitch_std'] = np.std(f0_valid)
            feats['pitch_range'] = np.max(f0_valid) - np.min(f0_valid)
        else:
            feats['pitch_mean'] = 0
            feats['pitch_std'] = 0
            feats['pitch_range'] = 0
    except:
        feats['pitch_mean'] = 0
        feats['pitch_std'] = 0
        feats['pitch_range'] = 0

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    feats['energy_mean'] = np.mean(rms)
    feats['energy_std'] = np.std(rms)
    feats['energy_range'] = np.max(rms) - np.min(rms)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats['zcr_mean'] = np.mean(zcr)
    feats['zcr_std'] = np.std(zcr)

    #Spectral Features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats['centroid_mean'] = np.mean(centroid)
    feats['centroid_std'] = np.std(centroid)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats['rolloff_mean'] = np.mean(rolloff)
    feats['rolloff_std'] = np.std(rolloff)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats['bandwidth_mean'] = np.mean(bandwidth)

    # Pause Detection
    intervals = librosa.effects.split(y, top_db=20)
    total_duration = len(y) / sr
    if len(intervals) > 0:
        speech_duration = sum((end - start) / sr for start, end in intervals)
        feats['silence_ratio'] = 1 - (speech_duration / total_duration)
        feats['num_pauses'] = len(intervals) - 1
    else:
        feats['silence_ratio'] = 1.0
        feats['num_pauses'] = 0

    #Onset Strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    feats['onset_mean'] = np.mean(onset_env)
    feats['onset_std'] = np.std(onset_env)

    #Audio duration
    feats['audio_duration'] = len(y) / sr

    return feats