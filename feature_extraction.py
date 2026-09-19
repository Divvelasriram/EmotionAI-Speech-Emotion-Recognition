import os
import librosa
import numpy as np

from config import (
    SAMPLE_RATE,
    DURATION,
    N_MFCC,
    N_MELS,
    EMOTIONS
)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(file_path):

    try:

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

        target_length = SAMPLE_RATE * DURATION

        if len(audio) < target_length:

            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )

        else:

            audio = audio[:target_length]

        return audio, sr

    except Exception as e:

        print("Audio loading error:", e)

        return None, None


# ============================================================
# EMOTION EXTRACTION
# ============================================================

def get_emotion(file_name):

    try:

        parts = file_name.split("-")

        emotion_code = parts[2]

        return EMOTIONS.get(emotion_code)

    except Exception:

        return None


# ============================================================
# STATISTICAL FEATURES
# ============================================================

def extract_statistical_features(file_path):

    audio, sr = load_audio(file_path)

    if audio is None:

        return None

    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y=audio
    )

    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # RMS Energy
    rms = librosa.feature.rms(
        y=audio
    )

    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


# ============================================================
# CNN + LSTM FEATURES
# ============================================================

def extract_deep_features(file_path):

    audio, sr = load_audio(file_path)

    if audio is None:

        return None

    # MFCC time-series
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Delta features
    delta = librosa.feature.delta(mfcc)

    # Delta-delta features
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Stack features
    features = np.vstack([
        mfcc,
        delta,
        delta2
    ])

    # Expected shape: (120, time_steps)
    return features.T.astype(np.float32)