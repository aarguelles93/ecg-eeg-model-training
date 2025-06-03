import numpy as np
import pandas as pd
import os

SAMPLES_DIR = 'temp'
os.makedirs(SAMPLES_DIR, exist_ok=True)

def save_csv(data, filename, header=True):
    df = pd.DataFrame(data)
    header_row = [f"c{i+1}" for i in range(df.shape[1])] if header else False
    df.to_csv(os.path.join(SAMPLES_DIR, filename), index=False, header=header_row)
    print(f"✅ Saved: {filename} (shape: {df.shape})")

# --- 1. ECG Sample: 187 points, repeated to 187x32 (simulated 32-channel ECG)
signal = 0.6 * np.sin(np.linspace(0, 2 * np.pi * 5, 187)) + 0.05 * np.random.randn(187)
ecg_matrix = np.tile(signal.reshape(-1, 1), (1, 32))
save_csv(ecg_matrix, "sample_ecg.csv", header=True)

# --- 2. EEG Sample: random 187x32 matrix (matches training shape)
eeg_matrix = np.random.randn(187, 32) * 0.3
save_csv(eeg_matrix, "sample_eeg_matrix.csv", header=True)

# --- 3. Flattened EEG: 32 channels × 128 timepoints → 1 row x 4096 columns
eeg_flat = (np.random.randn(32, 128) * 0.25).flatten()
save_csv([eeg_flat], "sample_eeg_flattened.csv", header=True)

print("\n📂 All sample files generated in:", os.path.abspath(SAMPLES_DIR))
