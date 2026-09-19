EmotionAI -- Speech Emotion Recognition 🎤

EmotionAI is an AI-powered Speech Emotion Recognition system that
analyzes audio recordings and predicts the emotion expressed in speech.

The project uses deep learning and machine learning models such as CNN,
LSTM, Random Forest, SVM, and MLP. A Streamlit web interface is included
for easy audio upload and emotion prediction.

🚀 Features

Upload audio files in WAV, MP3, OGG, and FLAC formats

Predict emotions from speech recordings

Use CNN and LSTM deep learning models

Compare Random Forest, SVM, and MLP models

Display predicted emotion and confidence score

View model performance and project information

Extract audio features using Librosa

🧠 Supported Emotions

The system is designed to recognize the following emotions:

Neutral

Calm

Happy

Sad

Angry

Fearful

Disgust

Surprised

🛠️ Technologies Used

Python

Streamlit

TensorFlow / Keras

Scikit-learn

Librosa

NumPy

Pandas

Matplotlib

Seaborn

Joblib

📂 Project Structure

EmotionAI-Speech-Emotion-Recognition/
│
├── app.py
├── config.py
├── emotion_recognition.py
├── feature_extraction.py
├── train_models.py
├── requirements.txt
│
├── models/
│   ├── cnn_model.keras
│   ├── lstm_model.keras
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── mlp.pkl
│   └── label_encoder.pkl
│
├── features/
│   └── deep_normalization.npz
│
├── outputs/
│   └── model_comparison.csv
│
└── README.md

Note: Model files, generated outputs, and dataset files may need to be
created by running the training pipeline.

📊 Dataset

This project uses the RAVDESS (Ryerson Audio-Visual Database of
Emotional Speech and Song) dataset.

The dataset should be placed in the following directory:

dataset/ravdess/

The complete dataset is not included in this repository. Download it
separately and place it in the required folder.

⚙️ Installation

1. Clone the repository

git clone https://github.com/your-username/EmotionAI-Speech-Emotion-Recognition.git
cd EmotionAI-Speech-Emotion-Recognition

2. Create a virtual environment

python -m venv venv

Activate the environment on Windows:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

▶️ Usage

Train the models

After placing the RAVDESS dataset in the required folder, run:

python train_models.py

This process trains the machine learning and deep learning models and
saves the generated files in the configured directories.

Run the Streamlit application

streamlit run app.py

The application will open in your browser. Upload an audio file and view
the predicted emotion.

📈 Model Types

The project includes the following models:

Random Forest

Support Vector Machine (SVM)

Multi-Layer Perceptron (MLP)

Convolutional Neural Network (CNN)

Long Short-Term Memory (LSTM)

Model performance can be reviewed through the Streamlit application's
model performance section and generated output files.

🔐 Notes

Ensure that the required trained model files are available before
running predictions.

Keep the dataset in the expected directory structure.

Check that the filenames referenced in app.py match the actual
model filenames.

Large datasets and trained model files may be excluded from GitHub
when necessary.

👨‍💻 Project Information

Project Name: EmotionAI -- Speech Emotion Recognition

Internship: CodeAlpha Machine Learning Internship

📜 License

This project is intended for educational and internship purposes.
