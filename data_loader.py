import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi(); api.authenticate()
# api.dataset_download_files('shayanfazeli/heartbeat', path='data/', unzip=True)


def load_ecg_data(filepath):
    # df = pd.read_csv(filepath, header=None)
    df = pd.read_csv(filepath, header=None, delimiter=',')
    X = df.iloc[:, :-1].values    # all but last column
    y = df.iloc[:, -1].values    # last column
    
    return X, y

def simulate_eeg_data(num_samples, signal_length=188):
    """Simulate random EEG-like signals (placeholder for real EEG)."""
    eeg_signals = np.random.randn(num_samples, signal_length) * 0.5  # Gaussian noise
    # labels = np.ones(num_samples) * 1  # Label 1 for EEG
    labels = np.ones(num_samples, dtype=int)
    return eeg_signals, labels

def load_eeg_data(filepath, channel_idx=0):
    # Now our train/test CSV *does* have a label column at the end
    df = pd.read_csv(filepath, header=0)

    # Ensure shape is consistent: 32 channels + 1 label column = 33 total
    if df.shape[1] != 33:
        raise ValueError(f"Expected 33 columns (32 channels + label), got {df.shape[1]}")
    
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()  # Drop rows with NaN values
    X = df.iloc[:, channel_idx].values.reshape(-1, 1)

    X = np.hstack([X] * 187)  # Repeat the channel data to match ECG length

    y = df.iloc[:,  -1].astype(int).values  # the injected “1” label
    
    return X, y

def normalize(X):
    """Normalize the data to the range [0, 1]."""
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min)

def prepare_dataset(ecg_path, eeg_path, channel_idx=0):
    print("Loading ECG data...")
    X_ecg, y_ecg = load_ecg_data(ecg_path)  

    print("Loading EEG data...")
    X_eeg, y_eeg = load_eeg_data(eeg_path, channel_idx=channel_idx)

    if X_ecg.shape[1] != X_eeg.shape[1]:
        raise ValueError(f"Feature mismatch: ECG {X_ecg.shape[1]} vs EEG {X_eeg.shape[1]}")
    
    y_ecg[:] = 0  # ECG label
    y_eeg[:] = 1  # EEG label

    X = np.vstack((X_ecg, X_eeg))
    y = np.concatenate((y_ecg, y_eeg))

    print("Shuffling data...")
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    print("Normalizing data...")
    X = normalize(X)
    X = X[..., np.newaxis]  # Add channel dimension

    print("Splitting data...", X.shape, y.shape)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def prepare_dataset_old(ecg_path, eeg_samples=1000):
    # 1) Load ECG
    X_ecg, y_ecg = load_ecg_data(ecg_path)
    n_features = X_ecg.shape[1]   # e.g. 188

    # 2) Simulate EEG with exactly the same length
    X_eeg, y_eeg = simulate_eeg_data(eeg_samples, signal_length=n_features)

    # 3) Relabel ECG→0, EEG→1
    y_ecg[:] = 0
    y_eeg[:] = 1

    # 4) Stack and shuffle
    X = np.vstack((X_ecg, X_eeg))
    y = np.concatenate((y_ecg, y_eeg))
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # 5) Normalize
    X = (X - X.min()) / (X.max() - X.min())

    # 6) Reshape for CNN
    X = X[..., np.newaxis]

    # 7) Split
    return train_test_split(X, y, test_size=0.2, random_state=42)
