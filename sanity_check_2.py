# sanity_check.py
import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

from data_loader import prepare_dataset

def plot_reliability_diagram(y_true, confidences, model_name='Model', results_dir='results'):
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, confidences, n_bins=10, strategy='uniform'
    )
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'{model_name} Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.title(f'Reliability Diagram - {model_name}')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(results_dir, f'calibration_{model_name.lower()}.png')
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


def evaluate_keras_model(model_path, X_test, y_test, model_name, flatten=False):
    print(f"\n🔍 Evaluating {model_name}...")

    if flatten:
        X_test = X_test.reshape(X_test.shape[0], -1)

    model = load_model(model_path)
    y_probs = model.predict(X_test)
    y_pred = (y_probs > 0.5).astype(int).flatten()

    # Ensure probabilities are flat (binary classification case)
    if y_probs.ndim > 1 and y_probs.shape[1] == 1:
        y_probs = y_probs.flatten()

    # Plot reliability
    confidences = y_probs
    plot_reliability_diagram(y_test, confidences, model_name=model_name)

    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("🧾 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ECG", "EEG"]))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def evaluate_svm(model_path, X_test, y_test):
    print("\n🔍 Evaluating SVM...")
    model = joblib.load(model_path)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred = model.predict(X_test_flat)

    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("🧾 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ECG", "EEG"]))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def evaluate_dual_branch(model_path, X_test, y_test):
    print("\n🔍 Evaluating Dual Branch CNN...")

    eeg_test = X_test[y_test == 1]
    ecg_test = X_test[y_test == 0]
    min_len = min(len(eeg_test), len(ecg_test))

    eeg_test = eeg_test[:min_len]
    ecg_test = ecg_test[:min_len]

    y_test_bal = np.concatenate([np.zeros(min_len), np.ones(min_len)])
    input_data = {
        'eeg_input': np.concatenate([eeg_test, np.zeros_like(ecg_test)]),
        'ecg_input': np.concatenate([np.zeros_like(eeg_test), ecg_test])
    }

    model = load_model(model_path)
    y_probs = model.predict(input_data)
    y_pred = (y_probs > 0.5).astype(int).flatten()

    print(f"✅ Accuracy: {accuracy_score(y_test_bal, y_pred):.4f}")
    print("🧾 Classification Report:")
    print(classification_report(y_test_bal, y_pred, target_names=["ECG", "EEG"]))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test_bal, y_pred))

def main():
    ecg_csv = os.path.join('data', 'mitbih_test.csv')
    eeg_csv = os.path.join('data', 'eeg_test.csv')
    X_train, X_test, y_train, y_test = prepare_dataset(ecg_csv, eeg_csv, downsample_ratio=1.5)

    model_dir = 'models'
    model_paths = {
        'svm': os.path.join(model_dir, 'svm_model.joblib'),
        'simple_cnn': os.path.join(model_dir, 'simple_cnn_final.keras'),
        'cnn_lstm': os.path.join(model_dir, 'cnn_lstm_final.keras'),
        'mlp': os.path.join(model_dir, 'mlp_final.keras'),
        'tcn': os.path.join(model_dir, 'tcn_final.keras'),
        'dual_branch': os.path.join(model_dir, 'dual_branch_final.keras')
    }

    evaluate_svm(model_paths['svm'], X_test, y_test)
    evaluate_keras_model(model_paths['simple_cnn'], X_test, y_test, 'Simple CNN')
    evaluate_keras_model(model_paths['cnn_lstm'], X_test, y_test, 'CNN + LSTM')
    evaluate_keras_model(model_paths['mlp'], X_test, y_test, 'MLP', flatten=True)
    evaluate_keras_model(model_paths['tcn'], X_test, y_test, 'TCN')
    evaluate_dual_branch(model_paths['dual_branch'], X_test, y_test)

if __name__ == "__main__":
    main()
