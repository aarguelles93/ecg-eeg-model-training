import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from scipy.special import softmax
from scipy.optimize import minimize

from data_loader import prepare_dataset

import joblib

# === Config ===
NUM_SAMPLES_PER_CLASS = 20
TRAIN_ECG_PATH = "data/mitbih_test.csv"
TRAIN_EEG_PATH = "data/eeg_test.csv"
MODEL_DIR = "models"
MODEL_NAMES = [
    ("svm_model.joblib", "svm"),
    ("mlp_best.keras", "mlp"),
    ("simple_cnn_best.keras", "cnn"),
    ("cnn_lstm_best.keras", "cnn_lstm"),
    ("tcn_best.keras", "tcn")
]
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def temperature_scale(logits, labels):
    """Optimizes temperature for calibration."""
    def nll_loss(temp):
        temp = temp[0]
        probs = softmax(logits / temp)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        correct_class_probs = probs[np.arange(len(labels)), labels]
        return -np.mean(np.log(correct_class_probs))

    res = minimize(nll_loss, [1.0], bounds=[(0.05, 10)])
    return res.x[0]

# def softmax(x, axis=1):
#     e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
#     return e_x / np.sum(e_x, axis=axis, keepdims=True)

def apply_temperature(logits, T):
    return logits / T


def plot_reliability_diagram(y_true, confidences, model_name='Model'):
    """Plots the reliability diagram for a set of predictions."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, confidences, n_bins=10, strategy='uniform'
    )

    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'{model_name} Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.title(f'Reliability Diagram - {model_name}')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'calibration_{model_name.lower()}.png'), bbox_inches="tight")
    plt.close()

# === Preprocessing (match your training pipeline) ===
def preprocess_signal(signal):
    signal = signal.astype(np.float32)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return signal
    return (signal - min_val) / (max_val - min_val)


def flatten_for_svm(x):
    return x.reshape(x.shape[0], -1)

def flatten_for_mlp(x):
    return x.reshape(x.shape[0], -1)

# === Load Samples ===
def load_samples(ecg_path, eeg_path, n_per_class, target_len=187):
    print("📥 Loading data...")
    # Load ECG
    ecg_df = pd.read_csv(ecg_path, header=None)
    ecg_df = ecg_df[ecg_df.iloc[:, -1] == 0]  # Keep only class 0
    ecg_samples = ecg_df.sample(n=n_per_class, random_state=42).iloc[:, :-1].values  # shape: (n, 187)
    
    # Preprocess ECG: reshape to (187, 32) by repeating last dimension
    ecg = np.array([
        preprocess_signal(row.reshape(target_len, 1).repeat(32, axis=1)) for row in ecg_samples
    ])

    # Load EEG
    eeg_df = pd.read_csv(eeg_path)
    if eeg_df.shape[1] != 33:
        raise ValueError(f"Expected 33 columns (32 channels + label), got {eeg_df.shape[1]}")

    eeg_df = eeg_df.apply(pd.to_numeric, errors='coerce').dropna()
    eeg_matrix = eeg_df.iloc[:, :32].values  # shape: (8064, 32)

    # Segment EEG into (187, 32)
    segments = []
    step = 20
    for start in range(0, len(eeg_matrix) - target_len + 1, step):
        segment = eeg_matrix[start:start + target_len]
        segments.append(segment)
        if len(segments) == n_per_class:
            break
    eeg_samples = np.array(segments)  # shape: (n_per_class, 187, 32)
    eeg = np.array([preprocess_signal(seg) for seg in eeg_samples])

    # Combine
    X = np.concatenate([ecg, eeg], axis=0)
    y = np.array([0] * len(ecg) + [1] * len(eeg))

    return X, y

def binary_temp_scale(logits, labels):
    def nll_loss(temp):
        temp = temp[0]
        probs = 1 / (1 + np.exp(-logits / temp))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    raw_loss = nll_loss([1.0])
    res = minimize(nll_loss, [1.0], bounds=[(0.05, 10.0)])
    best_temp = res.x[0]

    if res.fun < raw_loss:
        print(f"🔥 Applied temperature scaling with T = {best_temp:.3f} (NLL improved)")
        return best_temp
    else:
        print(f"⚠️ Skipped temperature scaling (no NLL improvement)")
        return 1.0

def multiclass_temp_scale(logits, labels):
    labels = np.array(labels)
    def nll_loss(temp):
        temp = temp[0]
        scaled_logits = logits / temp
        probs = softmax(scaled_logits, axis=1)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(np.log(probs[np.arange(len(labels)), labels]))

    raw_loss = nll_loss([1.0])
    res = minimize(nll_loss, [1.0], bounds=[(0.05, 10.0)])
    best_temp = res.x[0]

    if res.fun < raw_loss:
        print(f"🔥 Applied temperature scaling with T = {best_temp:.3f} (NLL improved)")
        return best_temp
    else:
        print(f"⚠️ Skipped temperature scaling (no NLL improvement)")
        return 1.0

# === Evaluation ===
def evaluate_model(name, model, X, y_true):
    print(f"\n🔍 Evaluating: {name.upper()}")
    print(f"  📏 Input shape: {X.shape}")
    print(f"  🔬 First sample (preview): {X[0].flatten()[:10]}")

    # if name == "svm":
    #     X_flat = flatten_for_svm(X)
    #     y_pred = model.predict(X_flat)
    #     probs = model.predict_proba(X_flat)
    # elif name == "mlp":
    #     X_flat = flatten_for_mlp(X)
    #     probs = model.predict(X_flat)
    #     if probs.shape[1] == 1:
    #         y_pred = (probs > 0.5).astype(int).flatten()
    #     else:
    #         y_pred = np.argmax(probs, axis=1)
    # else:
    #     probs = model.predict(X)
    #     if probs.shape[1] == 1:
    #         y_pred = (probs > 0.5).astype(int).flatten()
    #     else:
    #         y_pred = np.argmax(probs, axis=1)

    if name == "svm":
        X_flat = flatten_for_svm(X)
        y_pred = model.predict(X_flat)
        probs = model.predict_proba(X_flat)
    elif name == "mlp":
        X_flat = flatten_for_mlp(X)
        logits = model.predict(X_flat)
    else:
        logits = model.predict(X)

    # Apply temperature scaling (if not SVM)
    if name != "svm":
        if logits.shape[1] == 1:
            logits = logits.flatten()
            T = binary_temp_scale(logits, y_true)
            probs = 1 / (1 + np.exp(-logits / T))
            y_pred = (probs > 0.5).astype(int)
            probs = np.stack([1 - probs, probs], axis=1)
        else:
            T = multiclass_temp_scale(logits, y_true)
            scaled_logits = logits / T
            probs = softmax(scaled_logits, axis=1)
            y_pred = np.argmax(probs, axis=1)

    # Print predictions with confidence
    for i in range(len(y_true)):
        pred_label = int(y_pred[i])
        confidence = probs[i] if probs.shape[1] == 1 else probs[i][pred_label]
        confidence = float(confidence) if isinstance(confidence, np.ndarray) else confidence
        label = "ECG" if pred_label == 0 else "EEG"
        true_label = "ECG" if y_true[i] == 0 else "EEG"
        print(f"  [{i:02}] True: {true_label} | Predicted: {label} | Confidence: {confidence:.4f}")
    
    # Accuracy
    acc = np.mean(y_pred == y_true)
    print(f"✅ Accuracy on {len(y_true)} samples: {acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n📊 Confusion Matrix:")
    print(cm)
    print("\n🧾 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["ECG", "EEG"], zero_division=0))

    # Calibration plot
    confidences = [
        probs[i][int(y_pred[i])] if probs.ndim > 1 else probs[i]
        for i in range(len(y_pred))
    ]
    plot_reliability_diagram(y_true, confidences, model_name=name.upper())


# === Main ===
def main():
    DOWNSAMPLE_RATIO = 1.0  # Adjust as needed
    EEG_STEP = 50           # Adjust as needed
    print("downsample_ratio =", DOWNSAMPLE_RATIO, ", eeg_step =", EEG_STEP)
    X_train, X_test, y_train, y_test = prepare_dataset(
        ecg_path=TRAIN_ECG_PATH,
        eeg_path=TRAIN_EEG_PATH,
        downsample_ratio=DOWNSAMPLE_RATIO,
        eeg_step=EEG_STEP
    )

    X, y = X_test, y_test  # Sanity check runs on test split

    for model_file, name in MODEL_NAMES:
        model_path = os.path.join(MODEL_DIR, model_file)
        if name == "svm":
            model = joblib.load(model_path)
        else:
            model = load_model(model_path)
        evaluate_model(name, model, X, y)

if __name__ == "__main__":
    main()
