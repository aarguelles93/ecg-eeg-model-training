import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from tensorflow.keras.models import load_model
from scipy.special import softmax
from scipy.optimize import minimize
from data_loader import prepare_dataset
import joblib
import matplotlib.pyplot as plt

# === Config ===
TRAIN_ECG_PATH = "data/mitbih_train.csv"
TRAIN_EEG_PATH = "data/eeg_train.csv"
MODEL_DIR = "models"
RESULTS_DIR = 'results'
TARGET_RATIO = 1.0
EEG_STEP = 50
CALIBRATE = True

MODEL_NAMES = [
    ("svm_model.joblib", "svm"),
    ("mlp_best.keras", "mlp"),
    ("simple_cnn_best.keras", "cnn"),
    ("cnn_lstm_best.keras", "cnn_lstm"),
    ("tcn_best.keras", "tcn"),
    ("dual_branch_best.keras", "dual_branch")
]

os.makedirs(RESULTS_DIR, exist_ok=True)


# === Calibration Helpers ===
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
        print(f"🔥 Applied temperature scaling with T = {best_temp:.3f}")
        return best_temp
    else:
        print("⚠️ Skipped temperature scaling (no NLL improvement)")
        return 1.0


def multiclass_temp_scale(logits, labels):
    labels = np.array(labels)
    def nll_loss(temp):
        temp = temp[0]
        probs = softmax(logits / temp, axis=1)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.mean(np.log(probs[np.arange(len(labels)), labels]))

    raw_loss = nll_loss([1.0])
    res = minimize(nll_loss, [1.0], bounds=[(0.05, 10.0)])
    best_temp = res.x[0]
    if res.fun < raw_loss:
        print(f"🔥 Applied temperature scaling with T = {best_temp:.3f}")
        return best_temp
    else:
        print("⚠️ Skipped temperature scaling (no NLL improvement)")
        return 1.0


def plot_reliability_diagram(y_true, confidences, model_name='Model'):
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


# === Evaluation ===
def evaluate_model(name, model, X, y_true):
    print(f"\n🔍 Evaluating: {name.upper()}")
    print(f"  📏 Input shape: {X.shape}")

    if name == "svm":
        X_flat = X.reshape(X.shape[0], -1)
        y_pred = model.predict(X_flat)
        probs = model.predict_proba(X_flat)

    elif name == "mlp":
        X_flat = X.reshape(X.shape[0], -1)
        logits = model.predict(X_flat)

    elif name == "dual_branch":
        eeg_input = X[y_true == 1]
        ecg_input = X[y_true == 0]
        min_len = min(len(eeg_input), len(ecg_input))

        eeg_input = eeg_input[:min_len]
        ecg_input = ecg_input[:min_len]

        # Predict EEG (label=1)
        eeg_logits = model.predict({'eeg_input': eeg_input, 'ecg_input': ecg_input}).flatten()
        # Predict ECG (flip inputs, label=0)
        ecg_logits = model.predict({'eeg_input': ecg_input, 'ecg_input': eeg_input}).flatten()

        logits_all = np.concatenate([ecg_logits, eeg_logits])
        y_true = np.array([0] * min_len + [1] * min_len)

        if CALIBRATE:
            T = binary_temp_scale(logits_all, y_true)
        else:
            T = 1.0

        probs = 1 / (1 + np.exp(-logits_all / T))
        y_pred = (probs > 0.5).astype(int)
        probs = np.stack([1 - probs, probs], axis=1)

    else:
        logits = model.predict(X)

    if name not in ("svm", "dual_branch"):
        if logits.shape[1] == 1:
            logits = logits.flatten()
            T = binary_temp_scale(logits, y_true) if CALIBRATE else 1.0
            probs = 1 / (1 + np.exp(-logits / T))
            y_pred = (probs > 0.5).astype(int)
            probs = np.stack([1 - probs, probs], axis=1)
        else:
            T = multiclass_temp_scale(logits, y_true) if CALIBRATE else 1.0
            probs = softmax(logits / T, axis=1)
            y_pred = np.argmax(probs, axis=1)

    # Print predictions
    for i in range(len(y_pred)):
        label = "ECG" if y_pred[i] == 0 else "EEG"
        true_label = "ECG" if y_true[i] == 0 else "EEG"
        confidence = probs[i][y_pred[i]]
        print(f"  [{i:02}] True: {true_label} | Predicted: {label} | Confidence: {confidence:.4f}")

    acc = np.mean(y_pred == y_true)
    print(f"\n✅ Accuracy on {len(y_true)} samples: {acc*100:.2f}%")

    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\n🧾 Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["ECG", "EEG"],
        zero_division=0
    ))

    confidences = [probs[i][y_pred[i]] for i in range(len(y_pred))]
    plot_reliability_diagram(y_true, confidences, model_name=name.upper())


# === Main ===
def main():
    print("Preparing test dataset...")
    X_train, X_test, y_train, y_test = prepare_dataset(
        ecg_path=TRAIN_ECG_PATH,
        eeg_path=TRAIN_EEG_PATH,
        downsample_ratio=TARGET_RATIO,
        eeg_step=EEG_STEP
    )

    X, y = X_test, y_test

    for model_file, name in MODEL_NAMES:
        model_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_file}")
            continue

        print(f"\n📦 Loading model: {model_file}")
        model = joblib.load(model_path) if name == "svm" else load_model(model_path)
        evaluate_model(name, model, X, y)


if __name__ == "__main__":
    main()
