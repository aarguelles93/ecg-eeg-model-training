import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    accuracy_score, precision_score,
    recall_score, f1_score
)
from tensorflow.keras.models import load_model
from data_loader import prepare_dataset

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["ECG", "EEG"], yticklabels=["ECG", "EEG"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_roc_curve.png'))
    plt.close()

def save_metrics_csv(y_true, y_pred, model_name):
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0)
    }
    df = pd.DataFrame([metrics])
    file_path = os.path.join(RESULTS_DIR, f'{model_name}_metrics.csv')
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)

def print_evaluation(y_true, y_pred, model_name):
    print(f"\n📊 Evaluation for {model_name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["ECG", "EEG"]))
    plot_confusion_matrix(y_true, y_pred, model_name)
    save_metrics_csv(y_true, y_pred, model_name)

def evaluate_svm(X_test, y_test, model_path):
    print("\n🔍 Evaluating SVM...")
    svm = joblib.load(model_path)
    X_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred = svm.predict(X_flat)
    print_evaluation(y_test, y_pred, "SVM")

def evaluate_keras_model(model_path, X_test, y_test, name):
    print(f"\n🔍 Evaluating {name}...")
    model = load_model(model_path)
    y_probs = model.predict(X_test)
    y_pred = np.argmax(y_probs, axis=1)
    # y_pred = np.argmax(model.predict(X_test), axis=1)
    print_evaluation(y_test, y_pred, name)
    if y_probs.shape[1] == 2:
        plot_roc_curve(y_test, y_probs[:, 1], name)

def main():
    print("📦 Loading test data...")
    ecg_path = os.path.join('data', 'mitbih_train.csv')
    eeg_path = os.path.join('data', 'eeg_train.csv')

    _, X_test, _, y_test = prepare_dataset(ecg_path, eeg_path)

    model_dir = 'models'

    # Evaluate all models
    evaluate_svm(X_test, y_test, os.path.join(model_dir, 'svm_model.joblib'))
    evaluate_keras_model(os.path.join(model_dir, 'simple_cnn_best.keras'), X_test, y_test, "Simple CNN")
    evaluate_keras_model(os.path.join(model_dir, 'cnn_lstm_best.keras'), X_test, y_test, "CNN-LSTM")
    # evaluate_keras_model(os.path.join(model_dir, 'mlp_best.keras'), X_test, y_test, "MLP")
    # evaluate_keras_model(os.path.join(model_dir, 'mlp_tweaked_best.keras'), X_test, y_test, "MLP Tweaked")
    evaluate_keras_model(os.path.join(model_dir, 'mlp_salvaged_best.keras'), X_test, y_test, "MLP")
    evaluate_keras_model(os.path.join(model_dir, 'tcn_best.keras'), X_test, y_test, "TCN")

if __name__ == '__main__':
    main()