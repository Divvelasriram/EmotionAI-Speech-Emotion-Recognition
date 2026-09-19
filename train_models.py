import os
import warnings
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    LSTM,
    Dense,
    GlobalAveragePooling1D
)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    DATASET_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    FEATURE_DIR,
    EPOCHS,
    BATCH_SIZE,
    RANDOM_STATE,
    EMOTIONS
)

from feature_extraction import (
    extract_statistical_features,
    extract_deep_features,
    get_emotion
)

warnings.filterwarnings("ignore")

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset():

    statistical_features = []
    deep_features = []
    labels = []

    files = []

    for root, dirs, filenames in os.walk(DATASET_PATH):

        for file in filenames:

            if file.lower().endswith(".wav"):

                files.append(
                    os.path.join(root, file)
                )

    print("Total audio files:", len(files))

    for index, file_path in enumerate(
        tqdm(files, desc="Extracting features")
    ):

        file_name = os.path.basename(file_path)

        emotion = get_emotion(file_name)

        if emotion is None:

            continue

        stat_feature = extract_statistical_features(
            file_path
        )

        deep_feature = extract_deep_features(
            file_path
        )

        if stat_feature is None or deep_feature is None:

            continue

        statistical_features.append(stat_feature)

        deep_features.append(deep_feature)

        labels.append(emotion)

    X_stat = np.array(statistical_features)

    X_deep = np.array(deep_features)

    y = np.array(labels)

    print("Statistical feature shape:", X_stat.shape)

    print("Deep feature shape:", X_deep.shape)

    print("Labels shape:", y.shape)

    return X_stat, X_deep, y


# ============================================================
# 2. BUILD CNN MODEL
# ============================================================

def build_cnn(input_shape, num_classes):

    model = Sequential([

        Input(shape=input_shape),

        Conv1D(
            64,
            kernel_size=5,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling1D(pool_size=2),

        Dropout(0.3),

        Conv1D(
            128,
            kernel_size=5,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        MaxPooling1D(pool_size=2),

        Dropout(0.3),

        Conv1D(
            256,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        BatchNormalization(),

        GlobalAveragePooling1D(),

        Dense(128, activation="relu"),

        Dropout(0.4),

        Dense(
            num_classes,
            activation="softmax"
        )

    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 3. BUILD LSTM MODEL
# ============================================================

def build_lstm(input_shape, num_classes):

    model = Sequential([

        Input(shape=input_shape),

        LSTM(
            128,
            return_sequences=True
        ),

        Dropout(0.3),

        LSTM(64),

        Dropout(0.3),

        Dense(128, activation="relu"),

        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax"
        )

    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 4. TRAIN DEEP LEARNING MODELS
# ============================================================

def train_deep_models(X_deep, y_encoded, encoder):

    num_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_deep,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    # Normalize using training data only
    mean = X_train.mean(axis=(0, 1), keepdims=True)

    std = X_train.std(axis=(0, 1), keepdims=True)

    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std

    X_test = (X_test - mean) / std

    np.savez(
        os.path.join(FEATURE_DIR, "deep_normalization.npz"),
        mean=mean,
        std=std
    )

    input_shape = X_train.shape[1:]

    # CNN
    print("\nTraining CNN...")

    cnn = build_cnn(
        input_shape,
        num_classes
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.5
        )

    ]

    cnn.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    cnn.save(
        os.path.join(MODEL_DIR, "cnn_model.keras")
    )

    # LSTM
    print("\nTraining LSTM...")

    lstm = build_lstm(
        input_shape,
        num_classes
    )

    lstm.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    lstm.save(
        os.path.join(MODEL_DIR, "lstm_model.keras")
    )

    # Evaluation
    results = []

    for name, model in [
        ("CNN", cnn),
        ("LSTM", lstm)
    ]:

        predictions = np.argmax(
            model.predict(X_test, verbose=0),
            axis=1
        )

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

        results.append({

            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1

        })

        print(f"\n{name} Classification Report")

        print(
            classification_report(
                y_test,
                predictions,
                target_names=encoder.classes_
            )
        )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        plt.figure(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_
        )

        plt.title(f"{name} Confusion Matrix")

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{name.lower()}_confusion_matrix.png"
            )
        )

        plt.close()

    return results


# ============================================================
# 5. TRAIN CLASSICAL ML MODELS
# ============================================================

def train_classical_models(X_stat, y_encoded):

    X_train, X_test, y_train, y_test = train_test_split(
        X_stat,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "SVM": Pipeline([

            ("scaler", StandardScaler()),

            ("classifier", SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            ))

        ]),

        "MLP": Pipeline([

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

    for name, model in models.items():

        print("\nTraining:", name)

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

        results.append({

            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1

        })

        joblib.dump(
            model,
            os.path.join(
                MODEL_DIR,
                name.lower().replace(" ", "_") + ".pkl"
            )
        )

        print(
            classification_report(
                y_test,
                predictions
            )
        )

    return results


# ============================================================
# 6. MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 60)

    print("CODEALPHA SPEECH EMOTION RECOGNITION")

    print("=" * 60)

    X_stat, X_deep, y = load_dataset()

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    joblib.dump(
        encoder,
        os.path.join(
            MODEL_DIR,
            "label_encoder.pkl"
        )
    )

    np.save(
        os.path.join(FEATURE_DIR, "labels.npy"),
        y_encoded
    )

    classical_results = train_classical_models(
        X_stat,
        y_encoded
    )

    deep_results = train_deep_models(
        X_deep,
        y_encoded,
        encoder
    )

    results = pd.DataFrame(
        classical_results + deep_results
    )

    results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.csv"
        ),
        index=False
    )

    print("\nFINAL MODEL COMPARISON")

    print(results)

    plt.figure(figsize=(12, 6))

    plt.bar(
        results["Model"],
        results["Accuracy"]
    )

    plt.title("Model Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "model_comparison.png"
        )
    )

    plt.close()

    print("\nTraining completed successfully!")


if __name__ == "__main__":

    main()