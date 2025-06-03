import os
import joblib
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from data_loader import prepare_dataset
from models import (
    build_simple_cnn, build_cnn_lstm, build_svm_model, build_tcn,
    build_mlp, build_dual_branch_cnn
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

def save_training_curves(history, model_name, output_dir="results"):
    import datetime
    import matplotlib.pyplot as plt

    """Save training and validation accuracy/loss curves as PNG."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_training_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train Acc')
    plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

    return filepath

def print_evaluation(y_true, y_pred, model_name):
    print(f"\n Evaluation for {model_name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["ECG", "EEG"]))

def train_svm(X_train, y_train, X_test, y_test, output_dir):
    print("\n Training SVM...")
    # Flatten for SVM input
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    model = build_svm_model(y_train)
    model.fit(X_train_flat, y_train)

    y_pred = model.predict(X_test_flat)
    print_evaluation(y_test, y_pred, "SVM")

    joblib.dump(model, os.path.join(output_dir, 'svm_model.joblib'))
    print("✅ SVM model saved.\n")

def train_keras_model(model_builder, X_train, y_train, X_test, y_test, output_dir, name):
    print(f"\n Training {name}...")

    if name == 'mlp':
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test  = X_test.reshape(X_test.shape[0], -1)

    num_classes = len(np.unique(y_train))

    if num_classes == 2:
        y_train_cat = y_train  # Keep as 0/1
        loss_fn = 'binary_crossentropy'
    else:
        y_train_cat = to_categorical(y_train, num_classes)
        loss_fn = 'categorical_crossentropy'

    model = model_builder(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=loss_fn,
                  metrics=['accuracy'])

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    callbacks = [
        ModelCheckpoint(os.path.join(output_dir, f'{name}_best.keras'), save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2
    )
    save_training_curves(history, name, output_dir)

    # Evaluate
    if num_classes == 2:
        y_probs = model.predict(X_test)
        y_pred = (y_probs > 0.5).astype(int).flatten()
    else:
        y_probs = model.predict(X_test)
        y_pred = np.argmax(y_probs, axis=1)

    print_evaluation(y_test, y_pred, name)

    model.save(os.path.join(output_dir, f'{name}_final.keras'))
    print(f"✅ {name} model saved.\n")

def train_dual_branch(X_train, y_train, X_test, y_test, output_dir):
    print("\n🚀 Training redesigned Dual-Branch CNN (shared input)...")

    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    def save_training_curves(history, model_name='dual_branch'):
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history.get('accuracy', []), label='Train Acc')
        plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history.get('loss', []), label='Train Loss')
        plt.plot(history.history.get('val_loss', []), label='Val Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        filename = os.path.join(output_dir, f'{model_name}_training.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"📈 Training curves saved to: {filename}")

    def plot_confusion_matrix(y_true, y_pred, model_name='dual_branch'):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ECG', 'EEG'], yticklabels=['ECG', 'EEG'])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix: {model_name}")
        filename = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"📉 Confusion matrix saved to: {filename}")

    # === Model ===
    input_shape = X_train.shape[1:]  # (187, 32)
    model = build_dual_branch_cnn(input_shape=input_shape, num_classes=2)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # === Callbacks ===
    callbacks = [
        ModelCheckpoint(os.path.join(output_dir, 'dual_branch_best.keras'), save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]

    # === Training ===
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=2
    )

    save_training_curves(history, model_name='dual_branch')

    # === Evaluation ===
    print("\n🔍 Evaluating on test set...")
    y_probs = model.predict(X_test)
    y_pred = (y_probs > 0.5).astype(int).flatten()

    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ECG", "EEG"]))
    plot_confusion_matrix(y_test, y_pred)

    # === Save Final Model ===
    model.save(os.path.join(output_dir, 'dual_branch_final.keras'))
    print("✅ Dual-branch model training complete.\n")


def main(model_to_train='all'):
    print("Preparing dataset...")
    ecg_csv = os.path.join('data', 'mitbih_train.csv')
    eeg_path = os.path.join('data', 'eeg_train.csv')
    X_train, X_test, y_train, y_test = prepare_dataset(ecg_csv, eeg_path, downsample_ratio=1.5)

    print("X_train.shape =", X_train.shape, "y_train.shape =", y_train.shape)
    assert X_train.shape[1] == 187, f"Expected 187 timesteps, got {X_train.shape[1]}"
    assert X_train.shape[2] == 32, f"Expected 32 channels, got {X_train.shape[2]}"

    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)

    builders = {
        'svm': train_svm,
        'simple_cnn': build_simple_cnn,
        'cnn_lstm': build_cnn_lstm,
        'mlp': build_mlp,
        'tcn': build_tcn,
        'dual_branch': train_dual_branch
    }

    if model_to_train == 'all':
        train_svm(X_train, y_train, X_test, y_test, output_dir)
        for name, builder in builders.items():
            if name in ('svm', 'dual_branch'):
                if name == 'dual_branch':
                    builder(X_train, y_train, X_test, y_test, output_dir)
                continue
            train_keras_model(builder, X_train, y_train, X_test, y_test, output_dir, name)
        train_dual_branch(X_train, y_train, X_test, y_test, output_dir)
        print("✅ All models trained and saved.")
    elif model_to_train in builders:
        if model_to_train == 'svm':
            train_svm(X_train, y_train, X_test, y_test, output_dir)
        elif model_to_train == 'dual_branch':
            train_dual_branch(X_train, y_train, X_test, y_test, output_dir)
        else:
            train_keras_model(builders[model_to_train], X_train, y_train, X_test, y_test, output_dir, model_to_train)
        print(f"✅ {model_to_train} model trained and saved.")
    else:
        print(f"⚠️ Unknown model: '{model_to_train}'. No training performed.")

if __name__ == '__main__':
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    main(arg)
