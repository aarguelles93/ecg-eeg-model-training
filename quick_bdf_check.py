# Check first few files manually
import mne
import os

BDF_DIR = "data/bdf"
for i, filename in enumerate(sorted(os.listdir(BDF_DIR))):
    if filename.endswith(".bdf"):
        filepath = os.path.join(BDF_DIR, filename)
        raw = mne.io.read_raw_bdf(filepath, preload=True, verbose='ERROR')
        raw.pick("eeg")
        print(f"{filename}: {len(raw.ch_names)} EEG channels")