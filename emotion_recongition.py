# ============================================================
# CODEALPHA INTERNSHIP PROJECT
# EMOTION RECOGNITION FROM SPEECH
# ============================================================

import os
import warnings
import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = "ravdess"
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 3
OFFSET = 0

N_MFCC = 40
N_MELS = 128

RANDOM_STATE = 42


# ============================================================
# 2. EMOTION MAPPING
# ============================================================

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ============================================================
# 3. GET EMOTION FROM FILE NAME
# ============================================================

def get_emotion(file_name):

    try:

        parts = file_name.split("-")

        emotion_code = parts[2]

        emotion = EMOTIONS.get(emotion_code)

        return emotion

    except Exception:

        return None


# ============================================================
# 4. AUDIO LOADING
# ============================================================

def load_audio(file_path):

    try:

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION,
            offset=OFFSET
        )

        # Ensure fixed-length audio

        target_length = SAMPLE_RATE * DURATION

        if len(audio) < target_length:

            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )

        else:

            audio = audio[:target_length]

        return audio, sr

    except Exception as error:

        print("Audio loading error:", error)

        return None, None


# ============================================================
# 5. FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    audio, sr = load_audio(file_path)

    if audio is None:

        return None

    features = []

    # --------------------------------------------------------
    # A. MFCC FEATURES
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    mfcc_std = np.std(mfcc, axis=1)

    features.extend(mfcc_mean)

    features.extend(mfcc_std)

    # --------------------------------------------------------
    # B. MEL SPECTROGRAM
    # --------------------------------------------------------

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max
    )

    mel_mean = np.mean(mel_db, axis=1)

    mel_std = np.std(mel_db, axis=1)

    features.extend(mel_mean)

    features.extend(mel_std)

    # --------------------------------------------------------
    # C. CHROMA FEATURES
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    chroma_mean = np.mean(chroma, axis=1)

    chroma_std = np.std(chroma, axis=1)

    features.extend(chroma_mean)

    features.extend(chroma_std)

    # --------------------------------------------------------
    # D. ZERO CROSSING RATE
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    features.append(np.mean(zcr))

    features.append(np.std(zcr))

    # --------------------------------------------------------
    # E. RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )

    features.append(np.mean(rms))

    features.append(np.std(rms))

    # --------------------------------------------------------
    # FINAL FEATURE VECTOR
    # --------------------------------------------------------

    return np.array(features)


# ============================================================
# 6. LOAD DATASET
# ============================================================

def load_dataset():

    all_features = []

    all_labels = []

    file_names = []

    print("\nLoading RAVDESS dataset...\n")

    total_files = 0

    for root, directories, files in os.walk(DATASET_PATH):

        for file in files:

            if file.lower().endswith(".wav"):

                total_files += 1

    print("Total audio files:", total_files)

    processed = 0

    for root, directories, files in os.walk(DATASET_PATH):

        for file in files:

            if not file.lower().endswith(".wav"):

                continue

            file_path = os.path.join(root, file)

            emotion = get_emotion(file)

            if emotion is None:

                continue

            feature_vector = extract_features(file_path)

            if feature_vector is not None:

                all_features.append(feature_vector)

                all_labels.append(emotion)

                file_names.append(file)

            processed += 1

            if processed % 100 == 0:

                print(
                    f"Processed {processed}/{total_files}"
                )

    X = np.array(all_features)

    y = np.array(all_labels)

    print("\nFeature extraction completed!")

    print("Feature matrix shape:", X.shape)

    print("Label shape:", y.shape)

    return X, y


# ============================================================
# 7. SAVE DATASET FEATURES
# ============================================================

def save_features(X, y):

    np.save("features.npy", X)

    np.save("labels.npy", y)

    print("\nFeatures saved successfully!")


# ============================================================
# 8. TRAIN MODELS
# ============================================================

def train_models(X, y):

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    models = {

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),

        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=300,
                early_stopping=True,
                random_state=RANDOM_STATE
            ))
        ])

    }

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print("\n" + "=" * 60)

        print("Training:", model_name)

        print("=" * 60)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        print("Accuracy :", round(accuracy, 4))

        print("Precision:", round(precision, 4))

        print("Recall   :", round(recall, 4))

        print("F1 Score :", round(f1, 4))

        results.append({

            "Model": model_name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1

        })

        trained_models[model_name] = model

    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON\n")

    print(results_df)

    results_df.to_csv(
        "model_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # SELECT MODEL
    # --------------------------------------------------------

    best_model_name = results_df.loc[
        results_df["F1 Score"].idxmax(),
        "Model"
    ]

    best_model = trained_models[best_model_name]

    print("\nSelected model:", best_model_name)

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        os.path.join(MODEL_PATH, "emotion_model.pkl")
    )

    joblib.dump(
        encoder,
        os.path.join(MODEL_PATH, "label_encoder.pkl")
    )

    print("\nModel saved successfully!")

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    best_predictions = best_model.predict(X_test)

    print("\nCLASSIFICATION REPORT\n")

    print(
        classification_report(
            y_test,
            best_predictions,
            target_names=encoder.classes_
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        best_predictions
    )

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )

    plt.title("Emotion Recognition Confusion Matrix")

    plt.xlabel("Predicted Emotion")

    plt.ylabel("Actual Emotion")

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png"
    )

    plt.show()

    # --------------------------------------------------------
    # MODEL COMPARISON GRAPH
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    plt.title("Model Accuracy Comparison")

    plt.xlabel("Models")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "model_accuracy.png"
    )

    plt.show()

    return best_model, encoder


# ============================================================
# 9. PREDICT EMOTION FROM NEW AUDIO
# ============================================================

def predict_emotion(
    file_path,
    model,
    encoder
):

    feature_vector = extract_features(file_path)

    if feature_vector is None:

        return "Unable to process audio"

    feature_vector = feature_vector.reshape(1, -1)

    prediction = model.predict(
        feature_vector
    )

    predicted_emotion = encoder.inverse_transform(
        prediction
    )[0]

    return predicted_emotion


# ============================================================
# 10. MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print(" EMOTION RECOGNITION FROM SPEECH ")
    print(" CODEALPHA MACHINE LEARNING INTERNSHIP ")
    print("=" * 60)

    # Check dataset

    if not os.path.exists(DATASET_PATH):

        print("\nDataset folder not found!")

        print(
            "Please place the RAVDESS dataset in the project folder."
        )

        return

    # Load dataset

    X, y = load_dataset()

    if len(X) == 0:

        print("No audio features extracted!")

        return

    # Save features

    save_features(X, y)

    # Train models

    model, encoder = train_models(X, y)

    # Test audio prediction

    test_audio = "test_audio.wav"

    if os.path.exists(test_audio):

        emotion = predict_emotion(
            test_audio,
            model,
            encoder
        )

        print("\nPredicted Emotion:", emotion)

    else:

        print(
            "\nPlace test_audio.wav in the project folder "
            "to test predictions."
        )

    print("\nProject completed successfully!")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()