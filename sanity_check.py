import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# === Config ===
NUM_SAMPLES_PER_CLASS = 20
TRAIN_ECG_PATH = "data/mitbih_train.csv"
TRAIN_EEG_PATH = "data/eeg_train.csv"
MODEL_DIR = "models"
MODEL_NAMES = [
    ("svm_model.joblib", "svm"),
    ("mlp_best.keras", "mlp"),
    ("simple_cnn_best.keras", "cnn"),
    ("cnn_lstm_best.keras", "cnn_lstm"),
    ("tcn_best.keras", "tcn")
]

# === Preprocessing (match your training pipeline) ===
def preprocess_signal(signal):
    target_len = 187
    signal = signal.astype(np.float32)

    if signal.size > target_len:
        signal = signal[:target_len]
    elif signal.size < target_len:
        signal = np.pad(signal, (0, target_len - signal.size), mode="constant")

    # Normalize
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal.reshape((187, 1))  # fallback
    return ((signal - mean) / std).reshape((187, 1))

def flatten_for_svm(x):
    return x.reshape(x.shape[0], -1)

def flatten_for_mlp(x):
    return x.reshape(x.shape[0], -1)

# === Load Samples ===
def load_samples(ecg_path, eeg_path, n_per_class):
    print("📥 Loading data...")
    ecg_df = pd.read_csv(ecg_path).sample(n=n_per_class, random_state=42)
    eeg_df = pd.read_csv(eeg_path).sample(n=n_per_class, random_state=42)

    ecg_data = ecg_df.values
    eeg_data = eeg_df.values

    # Preprocess each sample individually
    ecg_data = np.array([preprocess_signal(x) for x in ecg_data])
    eeg_data = np.array([preprocess_signal(x) for x in eeg_data])

    X = np.concatenate([ecg_data, eeg_data], axis=0)
    y = np.array([0]*n_per_class + [1]*n_per_class)

    return X, y

# === Evaluation ===
def evaluate_model(name, model, X, y_true):
    print(f"\n🔍 Evaluating: {name.upper()}")
    print(f"  📏 Input shape: {X.shape}")
    print(f"  🔬 First sample (flattened preview): {X[0].flatten()[:10]}")

    if name == "svm":
        X_flat = flatten_for_svm(X)
        y_pred = model.predict(X_flat)
        probs = model.predict_proba(X_flat)
    elif name == "mlp":
        X_flat = flatten_for_mlp(X)
        probs = model.predict(X_flat)
        y_pred = np.argmax(probs, axis=1)
    else:
        probs = model.predict(X)
        y_pred = np.argmax(probs, axis=1)

    # Print predictions
    for i in range(len(y_true)):
        label = "ECG" if y_pred[i] == 0 else "EEG"
        true_label = "ECG" if y_true[i] == 0 else "EEG"
        confidence = probs[i][y_pred[i]]
        print(f"  [{i:02}] True: {true_label} | Predicted: {label} | Confidence: {confidence:.4f}")

    # Accuracy
    acc = np.mean(y_pred == y_true)
    print(f"✅ Accuracy on {len(y_true)} samples: {acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n📊 Confusion Matrix:")
    print(cm)
    print("\n🧾 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["ECG", "EEG"]))

# === Main ===
def main():
    X, y = load_samples(TRAIN_ECG_PATH, TRAIN_EEG_PATH, NUM_SAMPLES_PER_CLASS)

    for model_file, name in MODEL_NAMES:
        model_path = os.path.join(MODEL_DIR, model_file)
        if name == "svm":
            model = joblib.load(model_path)
        else:
            model = load_model(model_path)
        evaluate_model(name, model, X, y)

if __name__ == "__main__":
    main()
