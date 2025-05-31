import os
import joblib
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from data_loader import prepare_dataset
from models import (
    build_simple_cnn, build_cnn_lstm, build_svm_model, build_tcn,
    build_mlp
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

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
    y_train_cat = to_categorical(y_train, num_classes)
    # y_test_cat = to_categorical(y_test, num_classes)

    model = model_builder(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Calculate class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    callbacks = [
        ModelCheckpoint(os.path.join(output_dir, f'{name}_best.keras'), save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]

    model.fit(
        X_train, y_train_cat,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print_evaluation(y_test, y_pred, name)

    model.save(os.path.join(output_dir, f'{name}_final.keras'))
    print(f"✅ {name} model saved.\n")


def main(model_to_train='all'):
    print("Preparing dataset...")
    ecg_csv = os.path.join('data', 'mitbih_train.csv')
    eeg_path = os.path.join('data', 'eeg_train.csv')
    X_train, X_test, y_train, y_test = prepare_dataset(ecg_csv, eeg_path, channel_idx=0, downsample_ratio=2.0)

    print("X_train.shape =", X_train.shape, "y_train.shape =", y_train.shape)

    assert X_train.shape[1] == 187, f"Expected 187 timesteps, got {X_train.shape[1]}"
    assert X_train.shape[2] ==    1, f"Expected 1 channel,    got {X_train.shape[2]}"

    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)

    builders = {
        'svm': train_svm,
        'simple_cnn': build_simple_cnn,
        'cnn_lstm': build_cnn_lstm,
        'mlp': build_mlp,
        'tcn': build_tcn
    }

    if model_to_train == 'all':
        train_svm(X_train, y_train, X_test, y_test, output_dir)
        for name, builder in builders.items():
            if name == 'svm':
                continue
            train_keras_model(builder, X_train, y_train, X_test, y_test, output_dir, name)
        print("✅ All models trained and saved.")
    elif model_to_train in builders:
        if model_to_train == 'svm':
            train_svm(X_train, y_train, X_test, y_test, output_dir)
        else:
            train_keras_model(builders[model_to_train], X_train, y_train, X_test, y_test, output_dir, model_to_train)
        print(f"✅ {model_to_train} model trained and saved.")
    else:
        print(f"⚠️ Unknown model: '{model_to_train}'. No training performed.")

if __name__ == '__main__':
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    main(arg)
