import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from feature_extraction import (
    extract_deep_features,
    extract_statistical_features,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EmotionAI",
    page_icon="🎤",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .title {
        font-size: 45px;
        font-weight: bold;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #666;
    }

    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models"
FEATURE_DIR = "features"
OUTPUT_DIR = "outputs"


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_all_models():
    models = {}

    cnn_path = os.path.join(MODEL_DIR, "cnn_model.keras")
    lstm_path = os.path.join(MODEL_DIR, "lstm_model.keras")

    if os.path.exists(cnn_path):
        models["CNN"] = load_model(cnn_path)

    if os.path.exists(lstm_path):
        models["LSTM"] = load_model(lstm_path)

    for name in ["random_forest", "svm", "mlp"]:
        model_path = os.path.join(MODEL_DIR, name + ".pkl")

        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)

    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    normalization_path = os.path.join(
        FEATURE_DIR,
        "deep_normalization.npz",
    )

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            "label_encoder.pkl not found. Run train_models.py first."
        )

    if not os.path.exists(normalization_path):
        raise FileNotFoundError(
            "deep_normalization.npz not found. Run train_models.py first."
        )

    encoder = joblib.load(encoder_path)
    normalization = np.load(normalization_path)

    return models, encoder, normalization


# ============================================================
# DEEP LEARNING INPUT PREPARATION
# ============================================================

def prepare_deep_input(deep_feature, normalization, model):
    """
    Fixes the extra-dimension error:
    Incorrect: (1, 1, 130, 120)
    Correct:   (1, 130, 120)
    """

    deep_feature = np.asarray(deep_feature, dtype=np.float32)

    # Remove an unnecessary batch dimension if it already exists.
    if deep_feature.ndim == 3 and deep_feature.shape[0] == 1:
        deep_feature = deep_feature[0]

    if deep_feature.ndim != 2:
        raise ValueError(
            f"Unexpected deep feature shape: {deep_feature.shape}. "
            "Expected a 2D array such as (130, 120)."
        )

    # Convert normalization arrays to (120,) instead of (1, 1, 120).
    mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(normalization["std"], dtype=np.float32).reshape(-1)

    std = np.where(std == 0, 1.0, std)

    # Normalize without creating an unwanted extra dimension.
    deep_feature = (deep_feature - mean) / std

    # Read the expected input shape from the trained model.
    expected_shape = model.input_shape

    if isinstance(expected_shape, list):
        expected_shape = expected_shape[0]

    expected_steps = expected_shape[1]
    expected_features = expected_shape[2]

    if expected_features is not None:
        if deep_feature.shape[1] > expected_features:
            deep_feature = deep_feature[:, :expected_features]
        elif deep_feature.shape[1] < expected_features:
            feature_padding = np.zeros(
                (
                    deep_feature.shape[0],
                    expected_features - deep_feature.shape[1],
                ),
                dtype=np.float32,
            )
            deep_feature = np.concatenate(
                [deep_feature, feature_padding],
                axis=1,
            )

    if expected_steps is not None:
        if deep_feature.shape[0] > expected_steps:
            deep_feature = deep_feature[:expected_steps, :]
        elif deep_feature.shape[0] < expected_steps:
            time_padding = np.zeros(
                (
                    expected_steps - deep_feature.shape[0],
                    deep_feature.shape[1],
                ),
                dtype=np.float32,
            )
            deep_feature = np.concatenate(
                [deep_feature, time_padding],
                axis=0,
            )

    # Final shape should be (1, time_steps, features).
    return deep_feature.reshape(1, *deep_feature.shape)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🎤 EmotionAI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Speech Emotion Recognition System</div>',
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🎤 Emotion Prediction",
        "📊 Model Performance",
        "ℹ️ About Project",
    ],
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":
    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.header("Welcome to EmotionAI")

    st.write(
        """
        EmotionAI is a machine learning application that
        analyzes speech audio and predicts the emotion
        expressed in the recording.

        The system uses audio signal processing,
        machine learning, and deep learning techniques.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Deep Learning Models", "2")

    with col2:
        st.metric("Classical ML Models", "3")

    with col3:
        st.metric("Supported Emotions", "8")

    st.subheader("Technologies Used")

    st.write(
        """
        - Python
        - Librosa
        - TensorFlow / Keras
        - Scikit-learn
        - Streamlit
        - RAVDESS Dataset
        """
    )


# ============================================================
# EMOTION PREDICTION
# ============================================================

elif page == "🎤 Emotion Prediction":
    st.header("🎤 Upload Your Speech")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "ogg", "flac"],
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("🔍 Analyze Emotion"):
            temp_path = None

            try:
                with st.spinner("Analyzing speech..."):
                    file_extension = os.path.splitext(
                        uploaded_file.name
                    )[1].lower()

                    if file_extension not in [
                        ".wav",
                        ".mp3",
                        ".ogg",
                        ".flac",
                    ]:
                        file_extension = ".wav"

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=file_extension,
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    models, encoder, normalization = load_all_models()

                    deep_feature = extract_deep_features(temp_path)
                    stat_feature = extract_statistical_features(temp_path)

                    results = []

                    # -----------------------------
                    # CNN and LSTM predictions
                    # -----------------------------
                    if deep_feature is not None:
                        for name in ["CNN", "LSTM"]:
                            if name in models:
                                deep_input = prepare_deep_input(
                                    deep_feature,
                                    normalization,
                                    models[name],
                                )

                                probabilities = models[name].predict(
                                    deep_input,
                                    verbose=0,
                                )[0]

                                predicted_index = int(
                                    np.argmax(probabilities)
                                )

                                emotion = encoder.inverse_transform(
                                    [predicted_index]
                                )[0]

                                results.append(
                                    {
                                        "Model": name,
                                        "Emotion": emotion,
                                        "Confidence": float(
                                            probabilities[predicted_index]
                                        )
                                        * 100,
                                    }
                                )

                    # -----------------------------
                    # Classical ML predictions
                    # -----------------------------
                    if stat_feature is not None:
                        stat_input = np.asarray(
                            stat_feature,
                            dtype=np.float32,
                        ).reshape(1, -1)

                        for name in [
                            "random_forest",
                            "svm",
                            "mlp",
                        ]:
                            if name in models:
                                prediction = models[name].predict(
                                    stat_input
                                )[0]

                                emotion = encoder.inverse_transform(
                                    [prediction]
                                )[0]

                                confidence = None

                                if hasattr(
                                    models[name],
                                    "predict_proba",
                                ):
                                    probability = models[name].predict_proba(
                                        stat_input
                                    )[0]

                                    confidence = float(
                                        np.max(probability)
                                    ) * 100

                                results.append(
                                    {
                                        "Model": name.upper(),
                                        "Emotion": emotion,
                                        "Confidence": confidence,
                                    }
                                )

                    if results:
                        st.success("Emotion analysis completed!")

                        result_df = pd.DataFrame(results)

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                        )

                        st.subheader("Model Predictions")

                        for result in results:
                            confidence = result["Confidence"]

                            if confidence is not None:
                                st.write(
                                    f"**{result['Model']}**: "
                                    f"{result['Emotion'].upper()} "
                                    f"({confidence:.2f}%)"
                                )
                            else:
                                st.write(
                                    f"**{result['Model']}**: "
                                    f"{result['Emotion'].upper()}"
                                )
                    else:
                        st.error("No trained models found.")

            except Exception as error:
                st.error(f"Prediction error: {error}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":
    st.header("📊 Model Performance")

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv",
    )

    if os.path.exists(comparison_path):
        df = pd.read_csv(comparison_path)

        st.dataframe(
            df,
            use_container_width=True,
        )

        if "Model" in df.columns and "Accuracy" in df.columns:
            st.subheader("Accuracy Comparison")

            chart_data = df.set_index("Model")[["Accuracy"]]
            st.bar_chart(chart_data)
    else:
        st.warning("Train the models first to see performance.")


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":
    st.header("About EmotionAI")

    st.write(
        """
        EmotionAI is developed as part of the CodeAlpha
        Machine Learning Internship.

        The project uses speech signal processing and
        machine learning algorithms to classify emotions
        from audio recordings.

        Dataset:
        RAVDESS

        Deep Learning:
        CNN and LSTM

        Classical Machine Learning:
        Random Forest, SVM, MLP

        Feature Extraction:
        MFCC, Mel Spectrogram, Chroma, ZCR and RMS
        """
    )

    st.info(
        "Speech emotion predictions are estimates and should not "
        "be treated as definitive evidence of a person's actual "
        "emotional state."
    )
