import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi(); api.authenticate()
# api.dataset_download_files('shayanfazeli/heartbeat', path='data/', unzip=True)


def load_ecg_data(filepath, downsample_ratio=2.0, num_eeg=None):
    # df = pd.read_csv(filepath, header=None)
    df = pd.read_csv(filepath, header=None, delimiter=',')
    print(f"ECG data shape: {df.shape}")
    X = df.iloc[:, :-1].values    # all but last column
    y = df.iloc[:, -1].values    # last column

    # Only retain class 0 (normal) to match EEG vs ECG binary classification
    ecg_mask = y == 0
    X = X[ecg_mask]
    y = y[ecg_mask]

    # Downsample ECG to reduce imbalance (~80k to ~8k)
    if downsample_ratio is not None and num_eeg is not None:
        max_ecg = min(int(num_eeg * downsample_ratio), 12000)  # enforce 12k cap
        if X.shape[0] > max_ecg:
            indices = np.random.choice(X.shape[0], max_ecg, replace=False)
            X = X[indices]
            y = y[indices]

    return X, y

def simulate_eeg_data(num_samples, signal_length=188):
    """Simulate random EEG-like signals (placeholder for real EEG)."""
    eeg_signals = np.random.randn(num_samples, signal_length) * 0.5  # Gaussian noise
    # labels = np.ones(num_samples) * 1  # Label 1 for EEG
    labels = np.ones(num_samples, dtype=int)
    return eeg_signals, labels

def load_eeg_data(filepath, segment_length=187, step=2):
    # Now our train/test CSV *does* have a label column at the end
    df = pd.read_csv(filepath, header=0)

    # Ensure shape is consistent: 32 channels + 1 label column = 33 total
    if df.shape[1] != 33:
        raise ValueError(f"Expected 33 columns (32 channels + label), got {df.shape[1]}")
    
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    # signal = df.iloc[:, channel_idx].values  # shape: (8064,)
    signal_matrix = df.iloc[:, :32].values  # shape: (8064, 32)

    segments = []
    for start in range(0, len(signal_matrix) - segment_length + 1, step):
        segment = signal_matrix[start:start + segment_length]  # shape: (187, 32)
        segments.append(segment)
    print(f"EEG segments created: {len(segments)} with step={step}")

    X = np.array(segments)
    y = np.ones(len(X), dtype=int)  # Label 1 for EEG

    print(f"✅ EEG segments created: {len(X)} (shape: {X.shape})")

    return X, y

def normalize(X):
    """Normalize the data to the range [0, 1]."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

def normalize_per_sample(X):
    """Normalize each sample independently to [0, 1] over all values."""
    X_norm = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        sample = X[i]
        min_val = np.min(sample)
        max_val = np.max(sample)
        if max_val - min_val == 0:
            X_norm[i] = sample
        else:
            X_norm[i] = (sample - min_val) / (max_val - min_val)
    return X_norm

def prepare_dataset(ecg_path, eeg_path, downsample_ratio=2.0, eeg_step=2):
    print("Loading EEG data...")
    X_eeg, y_eeg = load_eeg_data(eeg_path, step=eeg_step)
    print(f"EEG data loaded: {X_eeg.shape} with labels {np.unique(y_eeg)}")

    print("Loading ECG data...")
    X_ecg, y_ecg = load_ecg_data(ecg_path, downsample_ratio=downsample_ratio, num_eeg=len(X_eeg))
    print(f"ECG data loaded: {X_ecg.shape} with labels {np.unique(y_ecg)}")

    if X_ecg.shape[1] != X_eeg.shape[1]:
        raise ValueError(f"Feature mismatch: ECG {X_ecg.shape[1]} vs EEG {X_eeg.shape[1]}")
    
    X_ecg = np.repeat(X_ecg[:, :, np.newaxis], 32, axis=2)  # shape: (N, 187, 32)

    y_ecg = np.zeros(len(X_ecg), dtype=int)  # explicitly recreate labels
    y_eeg = np.ones(len(X_eeg), dtype=int)

    assert set(y_ecg) == {0}, "ECG labels must be 0"
    assert set(y_eeg) == {1}, "EEG labels must be 1"

    print("📦 Stacking datasets...")
    X = np.vstack((X_ecg, X_eeg))
    y = np.concatenate((y_ecg, y_eeg))

    print("✅ Label check before shuffle:", np.bincount(y))
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    print("Class distribution after shuffle (ECG=0, EEG=1):", np.bincount(y))

    print("Normalizing data...")
    X = normalize_per_sample(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("✅ Label check in train:", np.bincount(y_train))
    print("✅ Label check in test:", np.bincount(y_test))
    return X_train, X_test, y_train, y_test

