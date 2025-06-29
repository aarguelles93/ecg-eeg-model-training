import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# --- Load the file ---
df = pd.read_csv("data/eeg_sample.csv")

# --- Flatten and preprocess it to shape (187, 1) ---
def preprocess(df: pd.DataFrame) -> np.ndarray:
    df = df.select_dtypes(include=[np.number])
    data = df.values.flatten()

    # Pad or trim
    target_len = 187
    if data.size > target_len:
        data = data[:target_len]
    elif data.size < target_len:
        data = np.pad(data, (0, target_len - data.size), mode="constant")

    # Normalize
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        raise ValueError("Zero variance in signal")
    data = (data - mean) / std

    return data.reshape((187, 1))

# Preprocess the input
input_data = preprocess(df)

# --- Predict with a model ---
# Choose model type: keras or svm
MODEL_PATH = "models/tcn_best.keras"  # or cnn_lstm_best.keras, etc.
model_type = "keras"  # or "svm"

if model_type == "keras":
    model = tf.keras.models.load_model(MODEL_PATH)
    input_batch = np.expand_dims(input_data, axis=0)  # shape (1, 187, 1)
    output = model.predict(input_batch)[0]
    label = "ECG" if np.argmax(output) == 0 else "EEG"
    print(f"[Keras] Predicted: {label} (Confidence: {np.max(output):.4f})")
    print("Raw Output:", output)

elif model_type == "svm":
    model = joblib.load(MODEL_PATH)
    flat_input = input_data.flatten().reshape(1, -1)
    output = model.predict_proba(flat_input)[0]
    label = "ECG" if np.argmax(output) == 0 else "EEG"
    print(f"[SVM] Predicted: {label} (Confidence: {np.max(output):.4f})")
    print("Raw Output:", output)
