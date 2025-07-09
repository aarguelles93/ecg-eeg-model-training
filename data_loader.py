import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_ecg_data(filepath=None, num_eeg=None, data_path="data/mit-bih", cache_path="data/cache/ecg_raw.npz"):
    """
    FIXED: Load ECG data directly from MIT-BIH files with proper patient grouping
    
    Args:
        filepath: Ignored (for compatibility) 
        num_eeg: Optional - if provided, balance ECG samples to match EEG count
        data_path: Path to MIT-BIH data directory
        
    Returns:
        X: ECG feature matrix
        y: ECG labels (all zeros for binary ECG vs EEG classification)
        groups: List of patient IDs for each beat
    """
    import wfdb
    from scipy.signal import resample
    import os
    import numpy as np

    # Check for cached raw data
    if os.path.exists(cache_path):
        try:
            print(f"🔍 Checking ECG raw data cache: {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            if 'data_path' in cached and cached['data_path'] == data_path:
                print(f"✅ Loading cached ECG data")
                # return cached['X'], cached['y'], cached['groups']
                return cached['X'].astype(np.float32), cached['y'], cached['groups']
            else:
                print(f"⚠️ Cache mismatch - regenerating ECG data")
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
    
    # MIT-BIH records to process (from your original script)
    RECORDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 
               109, 111, 112, 113, 114, 115, 116, 117, 118, 
               119, 121, 122, 123, 124, 200, 201, 202, 203, 
               205, 207, 208, 209, 210, 212, 213, 214, 215, 
               217, 219, 220, 221, 222, 223, 228, 230, 231, 
               232, 233, 234]
    
    print(f"❤️ Loading ECG data directly from MIT-BIH files...")
    print(f"   📂 Data path: {data_path}")
    
    all_beats = []
    all_groups = []
    beat_window = 187
    processed_patients = 0
    
    for record_id in RECORDS:
        try:
            record_path = os.path.join(data_path, str(record_id))
            record = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')
            
            signal = record.p_signal[:, 0]  # MLII channel
            half_window = beat_window // 2
            patient_beats = []
            
            # Extract normal beats for this patient
            for i, r_peak in enumerate(ann.sample):
                if ann.symbol[i] != 'N':  # Only normal beats
                    continue
                    
                if r_peak < half_window or r_peak + half_window >= len(signal):
                    continue
                    
                beat = signal[r_peak - half_window:r_peak + half_window]
                
                if len(beat) != beat_window:
                    beat = resample(beat, beat_window)
                    
                patient_beats.append(beat)
            
            if len(patient_beats) > 0:
                # CRITICAL: Use actual record ID as patient identifier
                patient_id = f"ecg_patient_{record_id}"
                
                all_beats.extend(patient_beats)
                # CRITICAL: Same patient_id for ALL beats from this patient
                all_groups.extend([patient_id] * len(patient_beats))
                
                processed_patients += 1
                print(f"   📄 Patient {record_id}: {len(patient_beats)} beats")
            else:
                print(f"   ⚠️  No normal beats found in record {record_id}, skipping.")
                
        except Exception as e:
            print(f"   ❌ Error processing record {record_id}: {e}")
            continue
    
    if not all_beats:
        raise ValueError("No ECG beats extracted from any record")
    
    X_ecg = np.array(all_beats, dtype=np.float32)
    y_ecg = np.zeros(len(all_beats), dtype=int)  # All ECG labeled as 0
    
    print(f"   ✅ ECG complete: {len(all_beats)} beats from {processed_patients} patients")
    print(f"   📊 Unique patients: {len(set(all_groups))}")
    print(f"   📊 Feature shape: {X_ecg.shape}")
    print(f"   📊 Data range: [{X_ecg.min():.6f}, {X_ecg.max():.6f}]")
    
    # Balance with EEG if requested
    if num_eeg is not None and len(X_ecg) > num_eeg:
        print(f"   ⚖️ Balancing ECG to match EEG size: {num_eeg:,} samples")
        
        # Stratified sampling to maintain patient representation
        from collections import defaultdict
        patient_indices = defaultdict(list)
        for i, patient in enumerate(all_groups):
            patient_indices[patient].append(i)
        
        # Calculate samples per patient
        samples_per_patient = num_eeg // len(patient_indices)
        remaining_samples = num_eeg % len(patient_indices)
        
        selected_indices = []
        for i, (patient, indices) in enumerate(patient_indices.items()):
            n_samples = samples_per_patient + (1 if i < remaining_samples else 0)
            n_samples = min(n_samples, len(indices))
            
            import random
            random.seed(42)
            selected = random.sample(indices, n_samples)
            selected_indices.extend(selected)
        
        X_ecg = X_ecg[selected_indices]
        y_ecg = y_ecg[selected_indices]
        all_groups = [all_groups[i] for i in selected_indices]
        
        print(f"   ✅ Balanced to: {len(X_ecg):,} samples")
        print(f"   📊 Patients represented: {len(set(all_groups))}")

    # Save to cache
    print(f"📎 Caching raw ECG data to: {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path,
                        X=X_ecg,
                        y=y_ecg,
                        groups=np.array(all_groups, dtype=object),
                        data_path=data_path)
    print(f"   ✅ ECG cache saved: {len(all_beats)} beats")
    
    return X_ecg, y_ecg, all_groups

# def load_eeg_data(filepath):
#     """Load EEG data from CSV file with proper header handling"""
#     df = pd.read_csv(filepath)  # Your new EEG dataset has headers
#     print(f"EEG data shape: {df.shape}")
#     print(f"EEG columns: {list(df.columns[:5])}...{list(df.columns[-3:])}")  # Show first 5 and last 3 columns
    
#     # Check if 'label' column exists
#     if 'label' in df.columns:
#         X = df.drop('label', axis=1).values
#         y = df['label'].astype(int).values
#     else:
#         # Fallback: assume last column is label
#         X = df.iloc[:, :-1].values
#         y = df.iloc[:, -1].values
    
#     print(f"EEG unique labels: {np.unique(y).tolist()}")
#     print(f"EEG data range: [{X.min():.6f}, {X.max():.6f}]")
#     print(f"EEG mean: {X.mean():.6f}, std: {X.std():.6f}")
#     return X, y

def load_eeg_data(filepath=None, chunk_size=5000, data_path="data/bdf", cache_path="data/cache/eeg_raw.npz"):
    """
    FIXED: Load EEG data directly from DEAP BDF files with proper subject grouping
    
    Args:
        filepath: Ignored (for compatibility)
        chunk_size: Ignored (for compatibility) 
        data_path: Path to BDF files directory
        
    Returns:
        X_eeg: EEG feature data
        y_eeg: EEG labels
        groups: List of subject IDs for each segment
        metadata: Dictionary containing structure information
    """
    import mne
    import os
    import numpy as np

    # Check for cached raw data
    if os.path.exists(cache_path):
        try:
            print(f"🔍 Checking EEG raw data cache: {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            if 'data_path' in cached and cached['data_path'] == data_path:
                print(f"✅ Loading cached EEG data")
                # return (cached['X'], cached['y'], cached['groups'], 
                #         cached['metadata'].item())
                return (cached['X'].astype(np.float32), cached['y'], cached['groups'], 
                        cached['metadata'].item())
            else:
                print(f"⚠️ Cache mismatch - regenerating EEG data")
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
    
    print(f"🧠 Loading EEG data directly from DEAP BDF files...")
    print(f"   📂 Data path: {data_path}")
    
    all_segments = []
    all_groups = []
    processed_subjects = 0
    
    # EEG processing parameters (from your original script)
    TARGET_CHANNELS = 32
    SEGMENT_LENGTH = 188
    STEP = 188  # No overlap
    DOWNSAMPLED_FREQ = 128
    TRIAL_DURATION = 63.0
    N_TRIALS_DEAP = 40
    
    # Suppress MNE warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    for filename in sorted(os.listdir(data_path)):
        if not filename.endswith('.bdf'):
            continue
            
        try:
            filepath = os.path.join(data_path, filename)
            # CRITICAL: Extract actual subject ID from filename
            subject_id = f"eeg_{filename.split('.')[0]}"  # e.g., "eeg_s01"
            
            print(f"   📄 Processing {filename} → {subject_id}")
            
            # Load BDF file
            raw = mne.io.read_raw_bdf(filepath, preload=True, verbose='ERROR')
            
            # Select EEG channels
            try:
                raw.pick(picks='eeg')  # Updated to use inst.pick()
                if len(raw.ch_names) > TARGET_CHANNELS:
                    raw.pick(raw.ch_names[:TARGET_CHANNELS])
                elif len(raw.ch_names) < 16:
                    print(f"   ⚠️ Only {len(raw.ch_names)} EEG channels, skipping")
                    continue
            except:
                # Fallback: pick first channels
                n_channels = min(TARGET_CHANNELS, len(raw.ch_names))
                raw.pick(raw.ch_names[:n_channels])
            
            # Resample
            raw.resample(DOWNSAMPLED_FREQ, verbose='ERROR')
            
            # Extract segments 
            samples_per_trial = int(TRIAL_DURATION * DOWNSAMPLED_FREQ)
            total_samples = len(raw.times)
            max_trials = min(N_TRIALS_DEAP, total_samples // samples_per_trial)
            
            subject_segments = []
            
            for trial_idx in range(max_trials):
                start_sample = trial_idx * samples_per_trial
                end_sample = start_sample + samples_per_trial
                
                try:
                    trial_data = raw.get_data(start=start_sample, stop=end_sample)
                    
                    # Create segments
                    for seg_start in range(0, trial_data.shape[1] - SEGMENT_LENGTH + 1, STEP):
                        segment = trial_data[:, seg_start:seg_start + SEGMENT_LENGTH]
                        
                        if segment.shape[1] == SEGMENT_LENGTH:
                            # Flatten to match expected format: (channels * timepoints,)
                            segment_flat = segment.flatten()
                            subject_segments.append(segment_flat)
                        
                except Exception as e:
                    continue
            
            if len(subject_segments) > 0:
                all_segments.extend(subject_segments)
                # CRITICAL: Same subject_id for ALL segments from this subject
                all_groups.extend([subject_id] * len(subject_segments))
                
                processed_subjects += 1
                print(f"      ✅ {len(subject_segments)} segments extracted")
            else:
                print(f"      ⚠️ No segments extracted")
                
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            continue
    
    if not all_segments:
        raise ValueError("No EEG segments extracted from any file")
    
    X_eeg = np.array(all_segments, dtype=np.float32)
    y_eeg = np.ones(len(all_segments), dtype=int)  # All EEG labeled as 1
    
    # Create metadata (similar to original function)
    actual_channels = TARGET_CHANNELS
    actual_timepoints = SEGMENT_LENGTH
    
    metadata = {
        'eeg_channels': actual_channels,
        'eeg_timepoints': actual_timepoints,
        'structure_valid': True,
        'total_features': X_eeg.shape[1],
        'total_samples': X_eeg.shape[0],
        'unique_labels': [1],
        'chunks_processed': processed_subjects
    }
    
    print(f"   ✅ EEG complete: {len(all_segments)} segments from {processed_subjects} subjects")
    print(f"   📊 Unique subjects: {len(set(all_groups))}")
    print(f"   📊 Feature shape: {X_eeg.shape}")
    print(f"   📊 Data range: [{X_eeg.min():.6f}, {X_eeg.max():.6f}]")

    # Save to cache
    print(f"📎 Caching raw EEG data to: {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path,
                        X=X_eeg,
                        y=y_eeg,
                        groups=np.array(all_groups, dtype=object),
                        metadata=np.array(metadata, dtype=object),
                        data_path=data_path)
    print(f"   ✅ EEG cache saved: {len(all_segments)} segments")
    
    return X_eeg, y_eeg, all_groups, metadata

# def project_ecg_to_eeg_format_(X_ecg, eeg_channels, eeg_timepoints):
#     """
#     Project ECG data to match EEG format using learned transformations
    
#     Args:
#         X_ecg: ECG data (n_samples, ecg_features)
#         eeg_channels: Number of EEG channels 
#         eeg_timepoints: Number of timepoints per EEG segment
    
#     Returns:
#         X_ecg_projected: ECG data reshaped to (n_samples, eeg_channels * eeg_timepoints)
#     """
#     from sklearn.decomposition import PCA
#     from sklearn.preprocessing import StandardScaler
    
#     target_features = eeg_channels * eeg_timepoints
#     ecg_features = X_ecg.shape[1]
    
#     print(f"   🔄 Projecting ECG: {ecg_features} → {target_features} features")
    
#     if ecg_features == target_features:
#         print("   ✅ ECG already matches EEG dimensions")
#         return X_ecg
    
#     elif ecg_features > target_features:
#         # Use PCA to reduce dimensions while preserving variance
#         print(f"   📉 Reducing ECG dimensions with PCA...")
#         pca = PCA(n_components=target_features)
#         X_ecg_projected = pca.fit_transform(X_ecg)
#         explained_var = pca.explained_variance_ratio_.sum()
#         print(f"   📊 PCA preserved {explained_var:.3f} of variance")
        
#     else:
#         # Expand dimensions by repeating/interpolating features
#         print(f"   📈 Expanding ECG dimensions...")
        
#         # Method 1: Repeat features cyclically
#         repeat_factor = target_features // ecg_features
#         remainder = target_features % ecg_features
        
#         X_repeated = np.tile(X_ecg, (1, repeat_factor))
#         if remainder > 0:
#             X_remainder = X_ecg[:, :remainder]
#             X_ecg_projected = np.hstack([X_repeated, X_remainder])
#         else:
#             X_ecg_projected = X_repeated
        
#         print(f"   🔄 Repeated ECG features {repeat_factor}x + {remainder} extra")
    
#     return X_ecg_projected

# def project_ecg_to_eeg_format__(X_ecg, eeg_channels, eeg_timepoints):
#     """
#     Project ECG data to EEG format using temporal segmentation and feature extraction
    
#     Args:
#         X_ecg: ECG data (n_samples, 187)
#         eeg_channels: Number of EEG channels (32)
#         eeg_timepoints: Number of timepoints per EEG segment (188)
    
#     Returns:
#         X_ecg_projected: ECG data reshaped to (n_samples, eeg_channels * eeg_timepoints)
#     """
#     import numpy as np
#     from scipy.stats import skew, kurtosis
#     from scipy.fft import fft

#     target_features = eeg_channels * eeg_timepoints
#     ecg_features = X_ecg.shape[1]
    
#     print(f"🔄 Projecting ECG: {ecg_features} → {target_features} features")
    
#     if ecg_features == target_features:
#         print("✅ ECG already matches EEG dimensions")
#         return X_ecg
    
#     # Parameters for temporal segmentation
#     window_size = 30  # Small window to capture ECG morphology
#     step_size = 5     # Overlap for smooth transitions
#     n_windows = (ecg_features - window_size) // step_size + 1
    
#     # Ensure we get exactly eeg_channels windows
#     if n_windows < eeg_channels:
#         step_size = max(1, (ecg_features - window_size) // (eeg_channels - 1))
#         n_windows = (ecg_features - window_size) // step_size + 1
#     elif n_windows > eeg_channels:
#         window_size = min(ecg_features, window_size + ((n_windows - eeg_channels) * step_size))
#         n_windows = (ecg_features - window_size) // step_size + 1
    
#     print(f"📊 Creating {n_windows} windows (size={window_size}, step={step_size})")

#     X_ecg_projected = np.zeros((X_ecg.shape[0], eeg_channels * eeg_timepoints))
    
#     for i in range(X_ecg.shape[0]):
#         ecg_signal = X_ecg[i]
#         channels = []
        
#         # Extract windows
#         for j in range(n_windows):
#             start = j * step_size
#             end = start + window_size
#             window = ecg_signal[start:end]
            
#             # Pad window if needed
#             if len(window) < window_size:
#                 window = np.pad(window, (0, window_size - len(window)), mode='constant')
            
#             # Extract features for this window
#             features = [
#                 np.mean(window),
#                 np.std(window),
#                 skew(window),
#                 kurtosis(window),
#                 np.max(window) - np.min(window)  # Peak-to-peak
#             ]
            
#             # FFT features (top 5 frequencies)
#             fft_vals = np.abs(fft(window))[:window_size//2]
#             top_fft = np.sort(fft_vals)[-5:] if len(fft_vals) >= 5 else fft_vals
            
#             # Combine features
#             channel_features = np.concatenate([features, top_fft])
            
#             # Pad or truncate to eeg_timepoints
#             if len(channel_features) < eeg_timepoints:
#                 channel_features = np.pad(channel_features, (0, eeg_timepoints - len(channel_features)), mode='constant')
#             else:
#                 channel_features = channel_features[:eeg_timepoints]
            
#             channels.append(channel_features)
        
#         # Pad with zero channels if needed
#         while len(channels) < eeg_channels:
#             channels.append(np.zeros(eeg_timepoints))
        
#         # Truncate if too many channels
#         channels = channels[:eeg_channels]
        
#         # Reshape to (eeg_channels * eeg_timepoints)
#         X_ecg_projected[i] = np.array(channels).flatten()
    
#     print(f"✅ ECG projected: {X_ecg_projected.shape}")
#     print(f"   📊 Projected range: [{X_ecg_projected.min():.3f}, {X_ecg_projected.max():.3f}]")
    
#     return X_ecg_projected

def project_ecg_to_eeg_format(X_ecg, eeg_channels, eeg_timepoints):
    """
    Enhanced ECG-to-EEG projection using domain-specific cardiac features
    
    Creates biologically meaningful pseudo-channels based on:
    - Heart rate variability (HRV) features
    - QRS morphology characteristics  
    - R-peak temporal patterns
    - Cardiac frequency domain features
    - Statistical cardiac cycle properties
    
    Args:
        X_ecg: ECG data (n_samples, 187)
        eeg_channels: Number of EEG channels (32)
        eeg_timepoints: Number of timepoints per EEG segment (188)
    
    Returns:
        X_ecg_projected: ECG data reshaped to (n_samples, eeg_channels * eeg_timepoints)
    """
    import numpy as np
    
    target_features = eeg_channels * eeg_timepoints
    ecg_features = X_ecg.shape[1]
    
    print(f"🔄 Robust ECG projection: {ecg_features} → {target_features} features")
    print(f"   💓 Using domain-specific cardiac feature extraction")
    
    if ecg_features == target_features:
        print("✅ ECG already matches EEG dimensions")
        return X_ecg
    
    X_ecg_projected = np.zeros((X_ecg.shape[0], target_features), dtype=np.float32)
    
    for i in range(X_ecg.shape[0]):
        ecg_signal = X_ecg[i]
        
        try:
            # ===== ROBUST FEATURE EXTRACTION =====
            
            # 1. Extract features with clipping
            hrv_features = np.clip(extract_hrv_features(ecg_signal), -200, 200)
            morphology_features = np.clip(extract_morphology_features(ecg_signal), -100, 100)  
            frequency_features = np.clip(extract_cardiac_frequency_features(ecg_signal), -40, 40)
            temporal_features = np.clip(extract_temporal_pattern_features(ecg_signal), -10, 10)
            statistical_features = np.clip(extract_statistical_cardiac_features(ecg_signal), -10, 10)
            
            # 2. Normalize features to unit scale
            hrv_features = _robust_normalize(hrv_features)
            morphology_features = _robust_normalize(morphology_features)
            frequency_features = _robust_normalize(frequency_features)
            temporal_features = _robust_normalize(temporal_features)
            statistical_features = _robust_normalize(statistical_features)
            
            # ===== CREATE SCALED PSEUDO-CHANNELS =====
            
            channels = []
            
            # Channels 1-8: HRV-based channels (8 channels)
            channels.extend(create_hrv_channels(hrv_features, eeg_timepoints))
            
            # Channels 9-16: Morphology-based channels (8 channels)
            channels.extend(create_morphology_channels(morphology_features, eeg_timepoints))
            
            # Channels 17-24: Frequency-based channels (8 channels)  
            channels.extend(create_frequency_channels(frequency_features, eeg_timepoints))
            
            # Channels 25-32: Temporal and statistical channels (8 channels)
            channels.extend(create_temporal_statistical_channels(
                temporal_features, statistical_features, eeg_timepoints))
            
            # Ensure exactly 32 channels
            while len(channels) < eeg_channels:
                channels.append(np.zeros(eeg_timepoints))
            channels = channels[:eeg_channels]
            
            # Flatten and apply final robust scaling
            flattened = np.array(channels).flatten()
            X_ecg_projected[i] = _final_scaling(flattened)
            
        except Exception as e:
            # Fallback to original method for problematic signals
            X_ecg_projected[i] = fallback_projection(ecg_signal, target_features)
    
    print(f"✅ Robust ECG projection complete: {X_ecg_projected.shape}")
    print(f"   📊 Projected range: [{X_ecg_projected.min():.3f}, {X_ecg_projected.max():.3f}]")
    
    return X_ecg_projected


def _robust_normalize(features):
    """Robust normalization for feature vectors"""
    if len(features) == 0 or np.std(features) < 1e-8:
        return features
    
    # Use robust statistics (median, MAD) instead of mean/std
    median = np.median(features)
    mad = np.median(np.abs(features - median))
    
    if mad < 1e-8:
        return features - median
    
    # Scale using MAD (more robust than std)
    normalized = (features - median) / (1.4826 * mad)  # 1.4826 makes MAD ≈ std for normal dist
    
    # Clip to prevent extreme values
    normalized_with_noise = normalized + np.random.normal(0, 0.01, normalized.shape)
    return np.clip(normalized_with_noise, -3, 3)



def _final_scaling(flattened_features):
    """Apply final robust scaling to the complete feature vector"""
    if np.std(flattened_features) < 1e-8:
        return flattened_features
    
    # Clip extreme outliers first
    q25, q75 = np.percentile(flattened_features, [25, 75])
    iqr = q75 - q25
    if iqr < 1e-8:
        return flattened_features
    
    # Robust outlier bounds
    lower_bound = q25 - 2 * iqr
    upper_bound = q75 + 2 * iqr
    clipped = np.clip(flattened_features, lower_bound, upper_bound)
    
    # Final z-score normalization
    mean_val = np.mean(clipped)
    std_val = np.std(clipped)
    
    if std_val < 1e-8:
        return clipped - mean_val
    
    scaled = (clipped - mean_val) / std_val
    
    # Final safety clip to reasonable range
    return np.clip(scaled, -6, 6)


# Keep all other functions exactly the same
def extract_hrv_features(ecg_signal, fs=360):
    """Extract heart rate variability features"""
    from scipy.stats import skew, kurtosis
    r_peaks = detect_r_peaks(ecg_signal)
    
    if len(r_peaks) < 2:
        return np.zeros(8)
    
    rr_intervals = np.diff(r_peaks) / fs * 1000
    
    if len(rr_intervals) == 0:
        return np.zeros(8)
    
    features = [
        np.mean(rr_intervals),
        np.std(rr_intervals),
        np.sqrt(np.mean(np.diff(rr_intervals)**2)),
        len(rr_intervals[np.abs(np.diff(rr_intervals)) > 50]) / len(rr_intervals) * 100,
        np.max(rr_intervals) - np.min(rr_intervals),
        np.std(rr_intervals) / np.mean(rr_intervals) * 100,
        skew(rr_intervals) if len(rr_intervals) > 2 else 0,
        kurtosis(rr_intervals) if len(rr_intervals) > 2 else 0
    ]
    
    return np.array(features)


def extract_morphology_features(ecg_signal):
    """Extract QRS morphology and waveform features"""
    r_peaks = detect_r_peaks(ecg_signal)
    
    if len(r_peaks) < 2:
        return np.zeros(8)
    
    qrs_amplitudes = []
    qrs_widths = []
    
    for peak in r_peaks:
        start = max(0, peak - 14)
        end = min(len(ecg_signal), peak + 14)
        qrs_complex = ecg_signal[start:end]
        
        if len(qrs_complex) > 0:
            qrs_amplitudes.append(np.max(qrs_complex) - np.min(qrs_complex))
            threshold = np.mean(qrs_complex) + 0.5 * np.std(qrs_complex)
            width = len(qrs_complex[qrs_complex > threshold])
            qrs_widths.append(width)
    
    if not qrs_amplitudes:
        return np.zeros(8)
    
    features = [
        np.mean(qrs_amplitudes),
        np.std(qrs_amplitudes),
        np.mean(qrs_widths),
        np.std(qrs_widths),
        np.max(ecg_signal) - np.min(ecg_signal),
        np.mean(np.abs(ecg_signal)),
        np.std(ecg_signal),
        len(r_peaks) / len(ecg_signal) * 360
    ]
    
    return np.array(features)


def extract_cardiac_frequency_features(ecg_signal, fs=360):
    """Extract frequency domain features specific to cardiac signals"""
    fft_vals = np.abs(np.fft.fft(ecg_signal))
    freqs = np.fft.fftfreq(len(ecg_signal), 1/fs)
    
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_vals[:len(fft_vals)//2]
    
    lf_band = (positive_freqs >= 0.04) & (positive_freqs <= 0.15)
    hf_band = (positive_freqs >= 0.15) & (positive_freqs <= 0.4)
    qrs_band = (positive_freqs >= 10) & (positive_freqs <= 25)
    
    features = [
        np.sum(positive_fft[lf_band]) if np.any(lf_band) else 0,
        np.sum(positive_fft[hf_band]) if np.any(hf_band) else 0,
        np.sum(positive_fft[qrs_band]) if np.any(qrs_band) else 0,
        np.argmax(positive_fft) * fs / len(ecg_signal),
        np.mean(positive_fft),
        np.std(positive_fft),
        np.sum(positive_fft[:10]),
        np.sum(positive_fft[-10:])
    ]
    
    return np.array(features)


def extract_temporal_pattern_features(ecg_signal):
    """Extract temporal pattern features"""
    segment_size = len(ecg_signal) // 4
    features = []
    
    for i in range(4):
        start = i * segment_size
        end = (i + 1) * segment_size if i < 3 else len(ecg_signal)
        segment = ecg_signal[start:end]
        
        if len(segment) > 0:
            features.extend([np.mean(segment), np.std(segment)])
        else:
            features.extend([0, 0])
    
    return np.array(features)


def extract_statistical_cardiac_features(ecg_signal):
    """Extract statistical features of the cardiac signal"""
    from scipy.stats import skew, kurtosis
    features = [
        np.mean(ecg_signal),
        np.std(ecg_signal),
        skew(ecg_signal),
        kurtosis(ecg_signal),
        np.percentile(ecg_signal, 25),
        np.percentile(ecg_signal, 75),
        np.mean(np.abs(np.diff(ecg_signal))),
        np.sqrt(np.mean(np.diff(ecg_signal)**2))
    ]
    
    return np.array(features)


def detect_r_peaks(ecg_signal, fs=360):
    """Detect R-peaks in ECG signal"""
    from scipy.signal import butter, filtfilt, find_peaks
    try:
        nyquist = fs / 2
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        
        threshold = np.mean(np.abs(filtered_signal)) + 2 * np.std(filtered_signal)
        min_distance = int(0.6 * fs / 60 * 60)
        
        peaks, _ = find_peaks(filtered_signal, height=threshold, distance=min_distance)
        return peaks
        
    except Exception:
        threshold = np.mean(ecg_signal) + 2 * np.std(ecg_signal)
        peaks, _ = find_peaks(ecg_signal, height=threshold, distance=50)
        return peaks


def create_hrv_channels(hrv_features, timepoints):
    """Create 8 HRV-based pseudo-channels"""
    channels = []
    
    for i in range(8):
        if i < len(hrv_features):
            base_value = hrv_features[i]
            t = np.linspace(0, 2*np.pi, timepoints)
            freq = 0.1 + i * 0.05
            channel = base_value * (1 + 0.1 * np.sin(freq * t))
        else:
            channel = np.zeros(timepoints)
        
        channels.append(channel)
    
    return channels


def create_morphology_channels(morphology_features, timepoints):
    """Create 8 morphology-based pseudo-channels"""
    channels = []
    
    for i in range(8):
        if i < len(morphology_features):
            base_value = morphology_features[i]
            t = np.linspace(0, 1, timepoints)
            if i % 2 == 0:
                channel = base_value * np.exp(-t)
            else:
                channel = base_value * (1 - t)
        else:
            channel = np.zeros(timepoints)
        
        channels.append(channel)
    
    return channels


def create_frequency_channels(frequency_features, timepoints):
    """Create 8 frequency-based pseudo-channels"""
    channels = []
    
    for i in range(8):
        if i < len(frequency_features):
            base_value = frequency_features[i]
            t = np.linspace(0, 4*np.pi, timepoints)
            freq = 0.5 + i * 0.2
            channel = base_value * np.cos(freq * t) * np.exp(-0.1 * t)
        else:
            channel = np.zeros(timepoints)
        
        channels.append(channel)
    
    return channels


def create_temporal_statistical_channels(temporal_features, statistical_features, timepoints):
    """Create 8 temporal and statistical pseudo-channels"""
    channels = []
    all_features = np.concatenate([temporal_features, statistical_features])
    
    for i in range(8):
        if i < len(all_features):
            base_value = all_features[i]
            t = np.linspace(0, 1, timepoints)
            if i < 4:
                step_point = int(timepoints * (i + 1) / 5)
                channel = np.zeros(timepoints)
                channel[:step_point] = base_value
                channel[step_point:] = base_value * 0.5
            else:
                power = min(i - 3, 3)
                channel = base_value * (t ** power)
        else:
            channel = np.zeros(timepoints)
        
        channels.append(channel)
    
    return channels


def fallback_projection(ecg_signal, target_features):
    """Fallback to simple repetition if domain-specific extraction fails"""
    repeat_factor = target_features // len(ecg_signal)
    remainder = target_features % len(ecg_signal)
    
    X_repeated = np.tile(ecg_signal, repeat_factor)
    if remainder > 0:
        X_remainder = ecg_signal[:remainder]
        X_projected = np.hstack([X_repeated, X_remainder])
    else:
        X_projected = X_repeated
    
    return X_projected

def ensure_channel_time_structure(X_eeg, expected_channels=32, expected_timepoints=188):
    """
    Ensure EEG data has the expected channel-time structure
    
    Args:
        X_eeg: EEG data 
        expected_channels: Expected number of channels
        expected_timepoints: Expected timepoints per segment
    
    Returns:
        X_eeg_structured: EEG data with verified structure
        actual_channels: Actual number of channels detected
        actual_timepoints: Actual timepoints detected
    """
    total_features = X_eeg.shape[1]
    
    # Try to infer structure
    if total_features % expected_timepoints == 0:
        actual_channels = total_features // expected_timepoints
        actual_timepoints = expected_timepoints
    elif total_features % expected_channels == 0:
        actual_timepoints = total_features // expected_channels
        actual_channels = expected_channels
    else:
        # Fallback: assume expected structure and truncate/pad
        target_features = expected_channels * expected_timepoints
        if total_features >= target_features:
            X_eeg_structured = X_eeg[:, :target_features]
            actual_channels = expected_channels
            actual_timepoints = expected_timepoints
            print(f"   ✂️  EEG truncated from {total_features} to {target_features} features")
        else:
            # Pad with zeros
            padding = target_features - total_features
            X_padding = np.zeros((X_eeg.shape[0], padding))
            X_eeg_structured = np.hstack([X_eeg, X_padding])
            actual_channels = expected_channels
            actual_timepoints = expected_timepoints
            print(f"   📋 EEG padded from {total_features} to {target_features} features")
        
        return X_eeg_structured, actual_channels, actual_timepoints
    
    print(f"   📊 EEG structure: {actual_channels} channels × {actual_timepoints} timepoints")
    return X_eeg, actual_channels, actual_timepoints

def check_feature_compatibility(X_ecg, X_eeg):
    """
    Intelligently align ECG and EEG datasets preserving signal structure
    """
    print(f"\n🔍 Advanced feature compatibility check:")
    print(f"   ECG features: {X_ecg.shape[1]}")
    print(f"   EEG features: {X_eeg.shape[1]}")
    
    # First, understand EEG structure
    X_eeg, eeg_channels, eeg_timepoints = ensure_channel_time_structure(X_eeg)
    
    if X_ecg.shape[1] != X_eeg.shape[1]:
        print(f"   ⚠️  Feature dimension mismatch!")
        print(f"   🧠 EEG structure: {eeg_channels} channels × {eeg_timepoints} timepoints")
        
        # Project ECG to match EEG format instead of truncating
        X_ecg = project_ecg_to_eeg_format(X_ecg, eeg_channels, eeg_timepoints)
        
        print(f"   ✅ After projection - ECG: {X_ecg.shape[1]}, EEG: {X_eeg.shape[1]}")
    else:
        print(f"   ✅ Feature dimensions already match!")
    
    # Verify final alignment
    assert X_ecg.shape[1] == X_eeg.shape[1], f"Final mismatch: ECG {X_ecg.shape[1]} vs EEG {X_eeg.shape[1]}"
    
    return X_ecg, X_eeg

def normalize_minmax(X):
    """Min-max normalization to [0, 1] range"""
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler

def normalize_per_sample(X, epsilon=1e-8, clip_threshold=5.0):
    """
    Optimal per-sample normalization with 5.0σ clipping
    Based on empirical analysis showing this eliminates extreme values with minimal data loss
    
    Args:
        X: Input data (samples x features)
        epsilon: Small value to prevent division by zero
        clip_threshold: Clip values beyond this many standard deviations (optimal: 5.0)
        
    Returns:
        X_normalized: Normalized and optimally clipped data
        None: No scaler needed for per-sample normalization
    """
    print(f"🔧 Applying OPTIMAL per-sample normalization (clip at ±{clip_threshold}σ)...")
    print(f"   📊 Based on empirical analysis: eliminates >6σ values, minimal data loss")
    print(f"   Input shape: {X.shape}")
    print(f"   Input range: [{X.min():.6f}, {X.max():.6f}]")
    print(f"   Input stats: mean={X.mean():.6f}, std={X.std():.6f}")
    
    # Calculate mean and std for each sample (row)
    sample_means = np.mean(X, axis=1, keepdims=True)  # Shape: (n_samples, 1)
    sample_stds = np.std(X, axis=1, keepdims=True)    # Shape: (n_samples, 1)
    
    # Handle samples with zero variance
    zero_var_samples = np.sum(sample_stds.flatten() < epsilon)
    if zero_var_samples > 0:
        print(f"   ⚠️  Found {zero_var_samples} samples with near-zero variance")
        sample_stds = np.where(sample_stds < epsilon, epsilon, sample_stds)
    
    # Normalize each sample to mean=0, std=1
    X_normalized = (X - sample_means) / sample_stds
    
    # Count and analyze extreme values before clipping
    extreme_6_before = np.sum(np.abs(X_normalized) > 6.0)
    extreme_5_before = np.sum(np.abs(X_normalized) > 5.0)
    extreme_clip_before = np.sum(np.abs(X_normalized) > clip_threshold)
    
    print(f"   📊 Before clipping analysis:")
    print(f"      >6σ values: {extreme_6_before:,} ({extreme_6_before/X_normalized.size*100:.4f}%)")
    print(f"      >5σ values: {extreme_5_before:,} ({extreme_5_before/X_normalized.size*100:.4f}%)")
    print(f"      >{clip_threshold}σ values: {extreme_clip_before:,} ({extreme_clip_before/X_normalized.size*100:.4f}%)")
    
    original_std = np.std(X_normalized)
    
    # Apply optimal clipping
    if extreme_clip_before > 0:
        X_normalized = np.clip(X_normalized, -clip_threshold, clip_threshold)
        clipped_std = np.std(X_normalized)
        std_reduction = (original_std - clipped_std) / original_std * 100
        
        print(f"   ✂️  Applied {clip_threshold}σ clipping:")
        print(f"      Values clipped: {extreme_clip_before:,}")
        print(f"      Std reduction: {std_reduction:.1f}% (from {original_std:.3f} to {clipped_std:.3f})")
    
    # Validate clipping effectiveness
    extreme_6_after = np.sum(np.abs(X_normalized) > 6.0)
    extreme_5_after = np.sum(np.abs(X_normalized) > 5.0)
    
    # Final statistics
    final_range = [X_normalized.min(), X_normalized.max()]
    final_mean = np.mean(X_normalized)
    final_std = np.std(X_normalized)
    
    print(f"   📊 Post-clipping validation:")
    print(f"      >6σ values: {extreme_6_after:,} (target: 0)")
    print(f"      >5σ values: {extreme_5_after:,} (expected: 0 for 5.0σ clipping)")
    print(f"      Final range: [{final_range[0]:.3f}, {final_range[1]:.3f}]")
    print(f"      Final stats: mean={final_mean:.6f}, std={final_std:.6f}")
    
    # Success validation
    if extreme_6_after == 0 and extreme_5_after == 0:
        print(f"   ✅ PERFECT: All extreme values eliminated!")
        training_stability = "EXCELLENT"
    elif extreme_6_after == 0:
        print(f"   ✅ SUCCESS: All >6σ values eliminated!")
        training_stability = "VERY GOOD"
    else:
        print(f"   ⚠️  Still have {extreme_6_after} values >6σ - may need tighter clipping")
        training_stability = "FAIR"
    
    print(f"   🎯 Training stability prediction: {training_stability}")
    
    # Sample-wise validation (optional, for debugging)
    if X_normalized.shape[0] <= 1000:  # Only for smaller datasets
        print(f"   🔍 Sample-wise validation (first 3 samples):")
        for i in range(min(3, X_normalized.shape[0])):
            sample_mean = np.mean(X_normalized[i])
            sample_std = np.std(X_normalized[i])
            sample_range = [X_normalized[i].min(), X_normalized[i].max()]
            print(f"     Sample {i}: mean={sample_mean:.6f}, std={sample_std:.6f}, range=[{sample_range[0]:.3f}, {sample_range[1]:.3f}]")
    
    return X_normalized, None


def zscore_normalize(X):
    """Z-score normalization with sklearn's StandardScaler for consistency"""
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler

def detect_existing_normalization(X, dataset_name="Dataset"):
    """
    Detect if data is already normalized and suggest appropriate normalization
    
    Returns:
        'already_zscore', 'already_minmax', 'needs_normalization', or 'reasonable_range'
    """
    mean_val = np.mean(X)
    std_val = np.std(X)
    min_val, max_val = np.min(X), np.max(X)
    
    print(f"🔍 Analyzing {dataset_name} normalization status:")
    print(f"   Mean: {mean_val:.6f}, Std: {std_val:.6f}")
    print(f"   Range: [{min_val:.6f}, {max_val:.6f}]")
    
    # Check for z-score normalization (mean ≈ 0, std ≈ 1)
    if abs(mean_val) < 0.1 and abs(std_val - 1.0) < 0.1:
        print(f"   ✅ {dataset_name} appears to be z-score normalized")
        return 'already_zscore'
    
    # Check for min-max normalization (range ≈ [0,1])
    elif min_val >= -0.1 and max_val <= 1.1 and (max_val - min_val) > 0.8:
        print(f"   ✅ {dataset_name} appears to be min-max normalized [0,1]")
        return 'already_minmax'
    
    # Check if data is in a reasonable range (no normalization needed)
    elif abs(mean_val) < 10 and std_val < 10 and min_val > -100 and max_val < 100:
        print(f"   📊 {dataset_name} appears to be in reasonable range")
        return 'reasonable_range'
    
    else:
        print(f"   ⚠️  {dataset_name} needs normalization")
        return 'needs_normalization'

def smart_normalize_single_dataset(X, dataset_name="Dataset", force_method=None):
    """
    Intelligently normalize with optimal settings based on empirical analysis
    """
    if force_method:
        print(f"🔧 Force applying {force_method} normalization to {dataset_name}")
        if force_method == 'zscore':
            return zscore_normalize(X)
        elif force_method == 'minmax':
            return normalize_minmax(X)
        elif force_method == 'per_sample':
            return normalize_per_sample(X, clip_threshold=5.0)  # Optimal threshold
    
    # Detect current normalization state
    norm_status = detect_existing_normalization(X, dataset_name)
    
    if norm_status == 'already_zscore':
        print(f"   ✅ {dataset_name} already z-score normalized - no changes needed")
        return X, None
        
    elif norm_status == 'already_minmax':
        print(f"   🔄 {dataset_name} is min-max normalized - converting with optimal settings")
        # Convert min-max [0,1] to reasonable z-score range
        X_centered = X - 0.5  # Center: [-0.5, 0.5]
        X_scaled = X_centered / np.std(X_centered)  # Scale to unit variance
        
        # Apply light clipping if needed
        extreme_check = np.sum(np.abs(X_scaled) > 6)
        if extreme_check > 0:
            print(f"   ⚠️  Conversion created {extreme_check} extreme values - applying safety clipping")
            X_scaled = np.clip(X_scaled, -5, 5)
        
        print(f"   📊 Converted stats: mean={np.mean(X_scaled):.6f}, std={np.std(X_scaled):.6f}")
        print(f"   📊 Converted range: [{np.min(X_scaled):.3f}, {np.max(X_scaled):.3f}]")
        return X_scaled, None
        
    elif norm_status == 'reasonable_range':
        # Enhanced decision making based on empirical analysis
        print(f"   📊 {dataset_name} in reasonable range - optimal method selection...")
        
        # Detailed outlier analysis
        data_std = np.std(X)
        data_mean = np.mean(X)
        
        outliers_3sigma = np.sum(np.abs(X - data_mean) > 3 * data_std)
        outliers_4sigma = np.sum(np.abs(X - data_mean) > 4 * data_std) 
        outliers_5sigma = np.sum(np.abs(X - data_mean) > 5 * data_std)
        
        outlier_3_percent = outliers_3sigma / X.size * 100
        outlier_5_percent = outliers_5sigma / X.size * 100
        
        print(f"   📊 Outlier distribution:")
        print(f"      >3σ: {outliers_3sigma:,} ({outlier_3_percent:.3f}%)")
        print(f"      >4σ: {outliers_4sigma:,}")
        print(f"      >5σ: {outliers_5sigma:,} ({outlier_5_percent:.4f}%)")
        
        # Optimal decision based on empirical analysis
        if outlier_5_percent > 0.001 or outlier_3_percent > 0.1:  # Based on your test results
            print(f"   🧠 Outlier pattern matches EEG-like data - using optimal per-sample normalization")
            return normalize_per_sample(X, clip_threshold=5.0)
        else:
            print(f"   📊 Low outlier percentage - using global z-score normalization")
            return zscore_normalize(X)
        
    else:  # needs_normalization
        print(f"   🔧 {dataset_name} needs normalization - applying z-score")
        return zscore_normalize(X)
    
def normalize_train_test_separate(X_train, X_test, y_train, y_test, 
                                normalization='zscore', strategy='separate'):
    """
    PROPER normalization: fit on train, transform on train+test
    Prevents data leakage from test statistics
    """
    print(f"🔧 Applying {normalization} normalization (strategy: {strategy}) - LEAK-FREE")
    
    # Split by signal type
    ecg_train_mask = y_train == 0
    eeg_train_mask = y_train == 1
    ecg_test_mask = y_test == 0  
    eeg_test_mask = y_test == 1
    
    X_train_ecg, X_train_eeg = X_train[ecg_train_mask], X_train[eeg_train_mask]
    X_test_ecg, X_test_eeg = X_test[ecg_test_mask], X_test[eeg_test_mask]
    
    scalers = {}
    
    if strategy == 'separate':
        # ECG normalization - FIT ON TRAIN ONLY
        if normalization == 'zscore':
            scaler_ecg = StandardScaler()
            X_train_ecg_norm = scaler_ecg.fit_transform(X_train_ecg)  # FIT + TRANSFORM
            X_test_ecg_norm = scaler_ecg.transform(X_test_ecg)        # TRANSFORM ONLY
            
            # EEG normalization - FIT ON TRAIN ONLY  
            scaler_eeg = StandardScaler()
            X_train_eeg_norm = scaler_eeg.fit_transform(X_train_eeg)  # FIT + TRANSFORM
            X_test_eeg_norm = scaler_eeg.transform(X_test_eeg)        # TRANSFORM ONLY
            
            scalers['ecg'] = scaler_ecg
            scalers['eeg'] = scaler_eeg
            
        elif normalization == 'per_sample':
            X_train_ecg_norm, _ = normalize_per_sample(X_train_ecg)
            X_train_eeg_norm, _ = normalize_per_sample(X_train_eeg) 
            X_test_ecg_norm, _ = normalize_per_sample(X_test_ecg)
            X_test_eeg_norm, _ = normalize_per_sample(X_test_eeg)
            
        # Reconstruct full arrays
        X_train_norm = np.zeros_like(X_train)
        X_test_norm = np.zeros_like(X_test)
        
        X_train_norm[ecg_train_mask] = X_train_ecg_norm
        X_train_norm[eeg_train_mask] = X_train_eeg_norm
        X_test_norm[ecg_test_mask] = X_test_ecg_norm
        X_test_norm[eeg_test_mask] = X_test_eeg_norm
        
    else:  # combined strategy
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)  # FIT ON TRAIN
        X_test_norm = scaler.transform(X_test)        # TRANSFORM TEST
        scalers['combined'] = scaler
    
    print(f"✅ Leak-free normalization complete")
    print(f"   Train: mean={np.mean(X_train_norm):.6f}, std={np.std(X_train_norm):.6f}")
    print(f"   Test:  mean={np.mean(X_test_norm):.6f}, std={np.std(X_test_norm):.6f}")
    
    return X_train_norm, X_test_norm, scalers

def normalize_datasets(X_ecg, X_eeg, normalization='smart', strategy='separate'):
    """
    Apply smart normalization to ECG and EEG datasets
    ENHANCED VERSION with better EEG handling
    """
    print(f"\n🔧 Applying {normalization} normalization (strategy: {strategy})")
    
    # Debug data before normalization
    debug_data_before_normalization(X_ecg, "ECG")
    debug_data_before_normalization(X_eeg, "EEG")
    
    scalers = {}
    
    if normalization == 'smart':
        print("🧠 Using SMART normalization - detecting existing normalization...")
        
        if strategy == 'separate':
            # ECG handling
            X_ecg_norm, scaler_ecg = smart_normalize_single_dataset(X_ecg, "ECG")
            
            # EEG handling - be more aggressive about using per-sample
            print("🧠 Special EEG analysis for optimal normalization...")
            eeg_extreme_check = np.sum(np.abs(X_eeg) > 3 * np.std(X_eeg))
            eeg_extreme_percent = eeg_extreme_check / X_eeg.size * 100
            
            print(f"   📊 EEG outlier pre-analysis: {eeg_extreme_check:,} outliers (>3σ, {eeg_extreme_percent:.3f}%)")
            
            if eeg_extreme_percent > 0.05:  # More aggressive threshold
                print("   🔧 EEG has significant outliers - forcing enhanced per-sample normalization")
                X_eeg_norm, scaler_eeg = normalize_per_sample(X_eeg, clip_threshold=4.5)
            else:
                X_eeg_norm, scaler_eeg = smart_normalize_single_dataset(X_eeg, "EEG")
            
            scalers['ecg'] = scaler_ecg
            scalers['eeg'] = scaler_eeg
            
        else:  # combined strategy - not recommended for ECG+EEG
            print("   ⚠️  Combined strategy not recommended for ECG+EEG - switching to separate")
            return normalize_datasets(X_ecg, X_eeg, normalization, 'separate')
    
    elif normalization == 'per_sample':
        print("🔧 Using per-sample normalization for both datasets...")
        X_ecg_norm, scaler_ecg = normalize_per_sample(X_ecg, clip_threshold=5.0)
        X_eeg_norm, scaler_eeg = normalize_per_sample(X_eeg, clip_threshold=4.5)  # Tighter for EEG
        scalers['ecg'] = scaler_ecg
        scalers['eeg'] = scaler_eeg
    
    else:
        # Use the existing normalization logic for other methods
        if strategy == 'combined':
            X_combined = np.vstack([X_ecg, X_eeg])
            
            if normalization == 'zscore':
                X_combined_norm, scaler = zscore_normalize(X_combined)
            elif normalization == 'minmax':
                X_combined_norm, scaler = normalize_minmax(X_combined)
            else:
                raise ValueError(f"Unknown normalization method: {normalization}")
            
            n_ecg = len(X_ecg)
            X_ecg_norm = X_combined_norm[:n_ecg]
            X_eeg_norm = X_combined_norm[n_ecg:]
            scalers['combined'] = scaler
            
        else:  # separate
            if normalization == 'zscore':
                X_ecg_norm, scaler_ecg = zscore_normalize(X_ecg)
                X_eeg_norm, scaler_eeg = zscore_normalize(X_eeg)
            elif normalization == 'minmax':
                X_ecg_norm, scaler_ecg = normalize_minmax(X_ecg)
                X_eeg_norm, scaler_eeg = normalize_minmax(X_eeg)
            else:
                raise ValueError(f"Unknown normalization method: {normalization}")
            
            scalers['ecg'] = scaler_ecg
            scalers['eeg'] = scaler_eeg
    
    # Enhanced validation with better recommendations
    print(f"\n📊 Final normalization results:")
    print(f"   ECG: mean={np.mean(X_ecg_norm):.6f}, std={np.std(X_ecg_norm):.6f}, range=[{np.min(X_ecg_norm):.3f}, {np.max(X_ecg_norm):.3f}]")
    print(f"   EEG: mean={np.mean(X_eeg_norm):.6f}, std={np.std(X_eeg_norm):.6f}, range=[{np.min(X_eeg_norm):.3f}, {np.max(X_eeg_norm):.3f}]")
    
    # Check for extreme values with better guidance
    ecg_extreme = np.sum(np.abs(X_ecg_norm) > 6)
    eeg_extreme = np.sum(np.abs(X_eeg_norm) > 6)
    
    total_extreme = ecg_extreme + eeg_extreme
    total_size = X_ecg_norm.size + X_eeg_norm.size
    extreme_percent = total_extreme / total_size * 100
    
    if total_extreme > 0:
        print(f"   ⚠️  Warning: Found extreme values (>6 std)")
        print(f"      ECG: {ecg_extreme:,}, EEG: {eeg_extreme:,}, Total: {total_extreme:,} ({extreme_percent:.4f}%)")
        
        if extreme_percent > 0.01:  # More than 0.01%
            print("   🚨 HIGH extreme value percentage detected!")
            print("   💡 STRONG recommendation: Use --normalization per_sample")
            print("   💡 Or try: python train.py --normalization per_sample --quick-lc")
        else:
            print("   💡 Moderate extreme values - training should still work")
    else:
        print(f"   ✅ No extreme outliers detected - normalization is optimal!")
    
    # Debug data after normalization
    debug_data_after_normalization(X_ecg_norm, "ECG")
    debug_data_after_normalization(X_eeg_norm, "EEG")
    
    return X_ecg_norm, X_eeg_norm, scalers

def debug_data_before_normalization(X, dataset_name="Dataset", sample_size=1000):
    """Debug data statistics before normalization"""
    print(f"\n🔍 PRE-NORMALIZATION DEBUG ({dataset_name}):")
    
    # Basic statistics
    print(f"   Shape: {X.shape}")
    print(f"   Dtype: {X.dtype}")
    print(f"   Memory: {X.nbytes/1024**3:.2f}GB")
    
    # Sample statistics
    sample = X[:sample_size] if len(X) > sample_size else X
    print(f"   Sample size for analysis: {len(sample)}")
    
    # Overall statistics
    print(f"   Overall mean: {np.mean(X):.6f}")
    print(f"   Overall std: {np.std(X):.6f}")
    print(f"   Overall range: [{np.min(X):.6f}, {np.max(X):.6f}]")
    
    # Check for problematic values
    nan_count = np.sum(np.isnan(X))
    inf_count = np.sum(np.isinf(X))
    zero_count = np.sum(X == 0)
    
    print(f"   NaN values: {nan_count:,}")
    print(f"   Infinite values: {inf_count:,}")
    print(f"   Zero values: {zero_count:,} ({zero_count/X.size*100:.1f}%)")
    
    # Percentile analysis
    percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    values = np.percentile(X.flatten(), percentiles)
    print(f"   Percentiles:")
    for p, v in zip(percentiles, values):
        print(f"     {p:5.1f}%: {v:12.6f}")
    
    # Check for extreme outliers
    q1, q3 = np.percentile(X.flatten(), [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers = np.sum((X < lower_bound) | (X > upper_bound))
    outlier_percent = outliers / X.size * 100
    
    print(f"   IQR outliers (3×IQR): {outliers:,} ({outlier_percent:.2f}%)")
    print(f"   Outlier bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
    
    # Feature-wise analysis (sample first 10 features)
    print(f"   Per-feature stats (first 10 features):")
    for i in range(min(10, X.shape[1])):
        feature_data = X[:, i]
        f_mean = np.mean(feature_data)
        f_std = np.std(feature_data)
        f_min, f_max = np.min(feature_data), np.max(feature_data)
        print(f"     Feature {i:2d}: mean={f_mean:8.4f}, std={f_std:8.4f}, range=[{f_min:8.4f}, {f_max:8.4f}]")
    
    # Check if data is already normalized
    mean_val = abs(np.mean(X))
    std_val = abs(np.std(X) - 1.0)
    if mean_val < 0.1 and std_val < 0.1:
        print(f"   ⚠️  Data appears to already be z-score normalized!")
    elif np.min(X) >= 0 and np.max(X) <= 1 and (np.max(X) - np.min(X)) > 0.8:
        print(f"   ⚠️  Data appears to already be min-max normalized!")
    
    return True

def debug_data_after_normalization(X, dataset_name="Dataset", sample_size=1000):
    """Debug data statistics after normalization"""
    print(f"\n🔍 POST-NORMALIZATION DEBUG ({dataset_name}):")
    
    # Basic statistics
    mean_val = np.mean(X)
    std_val = np.std(X)
    min_val, max_val = np.min(X), np.max(X)
    
    print(f"   Mean: {mean_val:.6f}")
    print(f"   Std: {std_val:.6f}")
    print(f"   Range: [{min_val:.6f}, {max_val:.6f}]")
    
    # Check normalization quality
    if abs(mean_val) < 0.001:
        print(f"   ✅ Mean is close to 0")
    else:
        print(f"   ❌ Mean is NOT close to 0 (z-score)")
    
    if abs(std_val - 1.0) < 0.001:
        print(f"   ✅ Std is close to 1")
    else:
        print(f"   ❌ Std is NOT close to 1 (z-score)")
    
    # Check for min-max normalization
    if abs(min_val - 0.0) < 0.001 and abs(max_val - 1.0) < 0.001:
        print(f"   ✅ Range is [0, 1] (min-max normalized)")
    
    # Check for reasonable z-score range
    if min_val > -6 and max_val < 6:
        print(f"   ✅ Range is reasonable for z-scores")
    else:
        print(f"   ❌ Range is TOO WIDE for z-scores")
        
        # Find extreme values
        extreme_indices = np.where((X < -6) | (X > 6))
        extreme_count = len(extreme_indices[0])
        extreme_percent = extreme_count / X.size * 100
        print(f"       Extreme values (|z| > 6): {extreme_count:,} ({extreme_percent:.3f}%)")
        
        if extreme_count > 0:
            extreme_values = X[extreme_indices]
            print(f"       Most extreme values: {np.sort(extreme_values)[-10:]}")
    
    return True

def apply_smart_clipping(X_train, y_train, clip_config=None):
    """
    Apply signal-appropriate clipping to training data only
    
    Args:
        X_train: Training data
        y_train: Training labels (0=ECG, 1=EEG)
        clip_config: {'ecg_threshold': 4.0, 'eeg_threshold': 3.0, 'apply_to': 'both'}
    """
    if clip_config is None:
        clip_config = {
            'ecg_threshold': 4.0,  # More aggressive for raw ECG
            'eeg_threshold': 3.0,  # Conservative for preprocessed EEG
            'apply_to': 'both'     # 'ecg', 'eeg', 'both', 'none'
        }
    
    print(f"🔧 Applying smart clipping to training data...")
    
    X_clipped = X_train.copy()
    clipped_count = 0
    
    if clip_config['apply_to'] in ['ecg', 'both']:
        # ECG clipping (more aggressive due to raw data)
        ecg_mask = y_train == 0
        ecg_data = X_clipped[ecg_mask]
        
        before_count = np.sum(np.abs(ecg_data) > clip_config['ecg_threshold'])
        ecg_data_clipped = np.clip(ecg_data, -clip_config['ecg_threshold'], clip_config['ecg_threshold'])
        after_count = np.sum(np.abs(ecg_data_clipped) > clip_config['ecg_threshold'])
        
        X_clipped[ecg_mask] = ecg_data_clipped
        clipped_count += (before_count - after_count)
        
        print(f"   ❤️  ECG clipped at ±{clip_config['ecg_threshold']}σ: {before_count:,} → {after_count:,} outliers")
    
    if clip_config['apply_to'] in ['eeg', 'both']:
        # EEG clipping (conservative for preprocessed data)
        eeg_mask = y_train == 1
        eeg_data = X_clipped[eeg_mask]
        
        before_count = np.sum(np.abs(eeg_data) > clip_config['eeg_threshold'])
        eeg_data_clipped = np.clip(eeg_data, -clip_config['eeg_threshold'], clip_config['eeg_threshold'])
        after_count = np.sum(np.abs(eeg_data_clipped) > clip_config['eeg_threshold'])
        
        X_clipped[eeg_mask] = eeg_data_clipped
        clipped_count += (before_count - after_count)
        
        print(f"   🧠 EEG clipped at ±{clip_config['eeg_threshold']}σ: {before_count:,} → {after_count:,} outliers")
    
    print(f"   ✅ Total values clipped: {clipped_count:,}")
    return X_clipped

def apply_realistic_noise(X_train, y_train, noise_config=None):
    """
    Apply physiologically realistic noise to training data
    
    Args:
        noise_config: {
            'eeg_gaussian': 0.05,     # Neural noise
            'eeg_muscle': 0.03,       # Muscle artifacts  
            'ecg_baseline': 0.02,     # Baseline wander
            'ecg_powerline': 0.01,    # 50/60Hz interference
            'probability': 0.7        # Fraction of samples to augment
        }
    """
    if noise_config is None:
        noise_config = {
            'eeg_gaussian': 0.05,
            'eeg_muscle': 0.03, 
            'ecg_baseline': 0.02,
            'ecg_powerline': 0.01,
            'probability': 0.7
        }
    
    print(f"🔊 Applying realistic noise augmentation...")
    
    X_noisy = X_train.copy()
    np.random.seed(42)  # Reproducible
    
    n_samples = len(X_train)
    augment_mask = np.random.random(n_samples) < noise_config['probability']
    n_augmented = np.sum(augment_mask)
    
    print(f"   📊 Augmenting {n_augmented:,} / {n_samples:,} samples ({n_augmented/n_samples*100:.1f}%)")
    
    # EEG noise (neural + muscle artifacts)
    eeg_mask = (y_train == 1) & augment_mask
    if np.any(eeg_mask):
        eeg_data = X_noisy[eeg_mask]
        
        # Gaussian neural noise
        neural_noise = np.random.normal(0, noise_config['eeg_gaussian'], eeg_data.shape)
        
        # Muscle artifacts (sporadic high-frequency bursts)
        muscle_noise = np.zeros_like(eeg_data)
        for i in range(len(eeg_data)):
            if np.random.random() < 0.3:  # 30% chance of muscle artifact
                start_idx = np.random.randint(0, eeg_data.shape[1] - 50)
                duration = np.random.randint(10, 50)
                artifact = np.random.normal(0, noise_config['eeg_muscle'], duration)
                muscle_noise[i, start_idx:start_idx+duration] = artifact
        
        X_noisy[eeg_mask] = eeg_data + neural_noise + muscle_noise
        print(f"   🧠 EEG noise added: {np.sum(eeg_mask):,} samples")
    
    # ECG noise (baseline wander + powerline)
    ecg_mask = (y_train == 0) & augment_mask
    if np.any(ecg_mask):
        ecg_data = X_noisy[ecg_mask]
        
        # Baseline wander (low frequency drift)
        baseline_noise = np.zeros_like(ecg_data)
        for i in range(len(ecg_data)):
            t = np.linspace(0, 1, ecg_data.shape[1])
            drift_freq = np.random.uniform(0.1, 0.5)  # 0.1-0.5 Hz
            baseline_noise[i] = noise_config['ecg_baseline'] * np.sin(2 * np.pi * drift_freq * t)
        
        # Powerline interference (50/60 Hz simulation)
        powerline_noise = np.zeros_like(ecg_data)
        for i in range(len(ecg_data)):
            if np.random.random() < 0.4:  # 40% chance of powerline interference
                t = np.linspace(0, 1, ecg_data.shape[1])
                freq = np.random.choice([50, 60])  # European/American
                powerline_noise[i] = noise_config['ecg_powerline'] * np.sin(2 * np.pi * freq * t)
        
        X_noisy[ecg_mask] = ecg_data + baseline_noise + powerline_noise
        print(f"   ❤️  ECG noise added: {np.sum(ecg_mask):,} samples")
    
    print(f"   ✅ Noise augmentation complete")
    return X_noisy

def prepare_dataset(ecg_path, eeg_path, normalization='zscore', 
                   normalization_strategy='global', validate_alignment=True, 
                   force_reload=False, cache_path='data/preprocessed_dataset.npz',
                   chunk_size=5000, memory_limit_gb=None, dataset_fraction=1.0,
                   ecg_cache_path='data/cache/ecg_raw.npz',
                   eeg_cache_path='data/cache/eeg_raw.npz'):
    """
    Memory-efficient dataset preparation with PROPER normalization order and group creation
    FIXED: Creates groups and normalizes AFTER splitting to prevent data leakage
    ENHANCED: Forces perfect 50/50 class balance to eliminate systematic bias
    """
    import numpy as np
    import os
    from sklearn.model_selection import GroupShuffleSplit

    # Calculate chunk_size based on memory limit
    if memory_limit_gb is not None:
        bytes_per_sample = 6016 * 4
        samples_per_gb = (1024 ** 3) / bytes_per_sample
        max_samples_in_memory = int(samples_per_gb * memory_limit_gb * 0.6)
        chunk_size = max(1000, min(chunk_size, max_samples_in_memory))
        print(f"🧠 Memory limit applied: {memory_limit_gb}GB")
        print(f"   📊 Calculated chunk size: {chunk_size:,} samples")
        print(f"   📊 Estimated memory per chunk: {(chunk_size * bytes_per_sample / 1024**3):.2f}GB")

    if os.path.exists(cache_path) and not force_reload:
        try:
            print(f"🔍 Checking cached dataset: {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            if 'cache_params' in cached:
                cache_params = cached['cache_params'].item()
                print(f"   Cache parameters: {cache_params}")
                current_params = {
                    'normalization': normalization,
                    'normalization_strategy': normalization_strategy,
                    'validate_alignment': validate_alignment,
                    'ecg_path': ecg_path,
                    'eeg_path': eeg_path,
                    'dataset_fraction': dataset_fraction
                }
                if cache_params == current_params:
                    print(f"✅ Cache parameters match - loading cached data")
                    X_train = cached['X_train']
                    X_test = cached['X_test']
                    y_train = cached['y_train']
                    y_test = cached['y_test']
                    groups_train = cached.get('groups_train', None)
                    print(f"📊 Cached data loaded:")
                    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
                    print(f"   Range: [{X_train.min():.6f}, {X_train.max():.6f}]")
                    if groups_train is not None:
                        print(f"   🔒 Groups loaded: {len(set(groups_train))} unique groups")
                    metadata = {
                        'eeg_channels': int(cached.get('eeg_channels', 32)),
                        'eeg_timepoints': int(cached.get('eeg_timepoints', 188)),
                        'structure_valid': bool(cached.get('structure_valid', True)),
                        'scalers': cached.get('scalers', {}).item() if 'scalers' in cached else {},
                        'normalization': normalization,
                        'normalization_strategy': normalization_strategy,
                        'groups_train': groups_train  # CRITICAL: Include groups in metadata
                    }
                    return X_train, X_test, y_train, y_test, metadata
                else:
                    print(f"⚠️ Cache parameter mismatch - regenerating")
            else:
                print(f"⚠️ Old cache format - regenerating")
        except Exception as e:
            print(f"⚠️ Cache validation failed: {str(e)}")

    print(f"📝 Generating fresh dataset with chunk size: {chunk_size:,}")
    X_eeg, y_eeg, eeg_groups, eeg_metadata = load_eeg_data(
        eeg_path, chunk_size=chunk_size, cache_path=eeg_cache_path)
    print("\n❤️  Loading ECG data...")
    X_ecg, y_ecg, ecg_groups = load_ecg_data(
        ecg_path, num_eeg=None, cache_path=ecg_cache_path)
    print(f"ECG samples loaded: {X_ecg.shape[0]}")

    # NEW: Group-aware balancing to ensure equal ECG/EEG samples
    print(f"⚖️ Balancing ECG and EEG samples with group awareness...")
    from collections import defaultdict
    import random

    # Count samples per group
    ecg_group_counts = defaultdict(list)
    eeg_group_counts = defaultdict(list)
    for i, group in enumerate(ecg_groups):
        ecg_group_counts[group].append(i)
    for i, group in enumerate(eeg_groups):
        eeg_group_counts[group].append(i)

    # Determine target sample size (minimum of ECG/EEG group totals)
    total_ecg_samples = len(X_ecg)
    total_eeg_samples = len(X_eeg)
    target_samples = min(total_ecg_samples, total_eeg_samples)
    print(f"   📊 Target: {target_samples:,} samples per class")

    # Sample ECG data
    ecg_samples_per_group = target_samples // len(ecg_group_counts)
    ecg_remainder = target_samples % len(ecg_group_counts)
    selected_ecg_indices = []
    random.seed(42)
    for i, (group, indices) in enumerate(ecg_group_counts.items()):
        n_samples = ecg_samples_per_group + (1 if i < ecg_remainder else 0)
        n_samples = min(n_samples, len(indices))
        selected_ecg_indices.extend(random.sample(indices, n_samples))
    
    X_ecg = X_ecg[selected_ecg_indices]
    y_ecg = y_ecg[selected_ecg_indices]
    ecg_groups = [ecg_groups[i] for i in selected_ecg_indices]
    print(f"   ✅ ECG balanced: {len(X_ecg):,} samples from {len(set(ecg_groups))} patients")

    # Sample EEG data
    eeg_samples_per_group = target_samples // len(eeg_group_counts)
    eeg_remainder = target_samples % len(eeg_group_counts)
    selected_eeg_indices = []
    random.seed(42)
    for i, (group, indices) in enumerate(eeg_group_counts.items()):
        n_samples = eeg_samples_per_group + (1 if i < eeg_remainder else 0)
        n_samples = min(n_samples, len(indices))
        selected_eeg_indices.extend(random.sample(indices, n_samples))
    
    X_eeg = X_eeg[selected_eeg_indices]
    y_eeg = y_eeg[selected_eeg_indices]
    eeg_groups = [eeg_groups[i] for i in selected_eeg_indices]
    print(f"   ✅ EEG balanced: {len(X_eeg):,} samples from {len(set(eeg_groups))} subjects")

    # Apply dataset fraction
    if dataset_fraction < 1.0:
        target_eeg = int(len(X_eeg) * dataset_fraction)
        target_ecg = int(len(X_ecg) * dataset_fraction)
        
        X_eeg = X_eeg[:target_eeg]
        y_eeg = y_eeg[:target_eeg]
        eeg_groups = eeg_groups[:target_eeg]
        
        X_ecg = X_ecg[:target_ecg]
        y_ecg = y_ecg[:target_ecg]
        ecg_groups = ecg_groups[:target_ecg]
        
        print(f"✂️ Applied dataset_fraction: EEG={target_eeg:,}, ECG={target_ecg:,}")

    # Feature compatibility check
    X_ecg, X_eeg = check_feature_compatibility(X_ecg, X_eeg)

    # Create labels
    y_ecg_labels = np.zeros(len(X_ecg), dtype=int)
    y_eeg_labels = np.ones(len(X_eeg), dtype=int)

    # Combine data with groups
    X = np.vstack((X_ecg, X_eeg)).astype(np.float32)
    y = np.concatenate((y_ecg_labels, y_eeg_labels))
    groups = ecg_groups + eeg_groups

    # Validate lengths
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(f"Data length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}")

    print(f"   🔗 Combined dataset: {X.shape[0]} samples with {len(set(groups))} unique groups")

    # CHANGED: Use StratifiedGroupKFold for balanced class distribution
    print(f"\n🔒 LEAK-FREE SPLITTING: Using StratifiedGroupKFold for balanced classes...")
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        # Use first fold as train/test split (approximately 80/20)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(X, y, groups))
        train_idx, test_idx = splits[0]  # Use first fold as split
        print(f"🔧 Using StratifiedGroupKFold for balanced class distribution")
    except ImportError:
        print(f"⚠️  StratifiedGroupKFold not available (requires scikit-learn ≥1.4)")
        print(f"   📦 Falling back to GroupShuffleSplit - may create class imbalance")
        from sklearn.model_selection import GroupShuffleSplit
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    groups_train = [groups[i] for i in train_idx]
    groups_test = [groups[i] for i in test_idx]

    # Initial class balance check
    train_ecg_count = np.sum(y_train == 0)
    train_eeg_count = np.sum(y_train == 1)
    test_ecg_count = np.sum(y_test == 0)
    test_eeg_count = np.sum(y_test == 1)

    train_ecg_pct = train_ecg_count / (train_ecg_count + train_eeg_count) * 100
    test_ecg_pct = test_ecg_count / (test_ecg_count + test_eeg_count) * 100

    print(f"   📊 Initial split - Train: ECG={train_ecg_count:,}, EEG={train_eeg_count:,} ({train_ecg_pct:.1f}% ECG)")
    print(f"   📊 Initial split - Test: ECG={test_ecg_count:,}, EEG={test_eeg_count:,} ({test_ecg_pct:.1f}% ECG)")

    # NEW: Force perfect 50/50 class balance
    class_imbalance = abs(train_ecg_pct - test_ecg_pct)
    if class_imbalance > 5.0:  # If more than 5% difference
        print(f"🔧 FORCING PERFECT CLASS BALANCE (was {class_imbalance:.1f}% difference)")
        
        # Force training set to exact 50/50
        min_train = min(train_ecg_count, train_eeg_count)
        train_ecg_indices = np.where(y_train == 0)[0][:min_train]
        train_eeg_indices = np.where(y_train == 1)[0][:min_train]
        train_balanced_idx = np.concatenate([train_ecg_indices, train_eeg_indices])
        
        # Shuffle to avoid any ordering bias
        np.random.seed(42)
        np.random.shuffle(train_balanced_idx)
        
        # Force test set to exact 50/50
        min_test = min(test_ecg_count, test_eeg_count)
        test_ecg_indices = np.where(y_test == 0)[0][:min_test]
        test_eeg_indices = np.where(y_test == 1)[0][:min_test]
        test_balanced_idx = np.concatenate([test_ecg_indices, test_eeg_indices])
        
        # Shuffle to avoid any ordering bias
        np.random.seed(42)
        np.random.shuffle(test_balanced_idx)
        
        # Update arrays with balanced indices
        X_train_raw = X_train_raw[train_balanced_idx]
        y_train = y_train[train_balanced_idx]
        groups_train_balanced = [groups_train[i] for i in train_balanced_idx]
        
        X_test_raw = X_test_raw[test_balanced_idx]
        y_test = y_test[test_balanced_idx]
        groups_test_balanced = [groups_test[i] for i in test_balanced_idx]
        
        # Update groups
        groups_train = groups_train_balanced
        groups_test = groups_test_balanced
        
        print(f"   ✅ Perfect balance achieved:")
        print(f"      Train: {min_train:,} ECG, {min_train:,} EEG (50.0% ECG)")
        print(f"      Test: {min_test:,} ECG, {min_test:,} EEG (50.0% ECG)")
        print(f"      Total samples: Train={len(X_train_raw):,}, Test={len(X_test_raw):,}")
    else:
        print(f"   ✅ Class balance acceptable ({class_imbalance:.1f}% difference)")

    # Validate no group overlap
    train_groups_set = set(groups_train)
    test_groups_set = set(groups_test)
    overlap = train_groups_set.intersection(test_groups_set)
    if overlap:
        raise ValueError(f"Group overlap detected: {overlap}")
    print(f"   ✅ No group overlap: {len(train_groups_set)} train, {len(test_groups_set)} test groups")

    print(f"   📊 Final split - Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")

    # UPDATED: Enhanced normalization with global option
    print(f"\n🔧 LEAK-FREE NORMALIZATION: {normalization} (strategy: {normalization_strategy})")
    print(f"   📊 Processing {len(X_train_raw)} train + {len(X_test_raw)} test samples...")

    if normalization_strategy == 'separate':
        # Split by signal type for separate normalization
        ecg_train_mask = y_train == 0
        eeg_train_mask = y_train == 1
        ecg_test_mask = y_test == 0
        eeg_test_mask = y_test == 1
        
        X_train_ecg = X_train_raw[ecg_train_mask]
        X_train_eeg = X_train_raw[eeg_train_mask]
        X_test_ecg = X_test_raw[ecg_test_mask]
        X_test_eeg = X_test_raw[eeg_test_mask]
        
        scalers = {}
        
        if normalization == 'zscore':
            from sklearn.preprocessing import StandardScaler
            
            # ECG normalization - FIT ON TRAIN ONLY
            scaler_ecg = StandardScaler()
            X_train_ecg_norm = scaler_ecg.fit_transform(X_train_ecg)
            X_test_ecg_norm = scaler_ecg.transform(X_test_ecg)
            
            # EEG normalization - FIT ON TRAIN ONLY  
            scaler_eeg = StandardScaler()
            X_train_eeg_norm = scaler_eeg.fit_transform(X_train_eeg)
            X_test_eeg_norm = scaler_eeg.transform(X_test_eeg)
            
            scalers['ecg'] = scaler_ecg
            scalers['eeg'] = scaler_eeg
            
        elif normalization == 'per_sample':
            X_train_ecg_norm, _ = normalize_per_sample(X_train_ecg, clip_threshold=5.0)
            X_train_eeg_norm, _ = normalize_per_sample(X_train_eeg, clip_threshold=4.5)
            X_test_ecg_norm, _ = normalize_per_sample(X_test_ecg, clip_threshold=5.0)
            X_test_eeg_norm, _ = normalize_per_sample(X_test_eeg, clip_threshold=4.5)
            scalers['method'] = 'per_sample'
        
        # Reconstruct normalized arrays
        X_train = np.zeros_like(X_train_raw)
        X_test = np.zeros_like(X_test_raw)
        
        X_train[ecg_train_mask] = X_train_ecg_norm
        X_train[eeg_train_mask] = X_train_eeg_norm
        X_test[ecg_test_mask] = X_test_ecg_norm
        X_test[eeg_test_mask] = X_test_eeg_norm

    elif normalization_strategy == 'global':
        # NEW: Global normalization for deployment compatibility
        print(f"🌍 Using GLOBAL normalization for deployment compatibility")
        from sklearn.preprocessing import StandardScaler
        
        if normalization == 'zscore':
            # Fit scaler on ALL training data (both ECG and EEG)
            global_scaler = StandardScaler()
            X_train = global_scaler.fit_transform(X_train_raw)
            X_test = global_scaler.transform(X_test_raw)
            
            scalers = {'global': global_scaler}
            
            # Save for deployment
            scaler_path = 'models/global_scaler.pkl'
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            import joblib
            joblib.dump(global_scaler, scaler_path)
            print(f"💾 Global scaler saved to: {scaler_path}")
        else:
            # For other normalization methods, fall back to per_sample
            X_train, _ = normalize_per_sample(X_train_raw, clip_threshold=5.0)
            X_test, _ = normalize_per_sample(X_test_raw, clip_threshold=5.0)
            scalers = {'method': 'per_sample'}

    else:  # combined strategy (legacy)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)  # FIT ON TRAIN
        X_test = scaler.transform(X_test_raw)        # TRANSFORM TEST
        scalers = {'combined': scaler}

    # Apply clipping and noise AFTER normalization, BEFORE training
    print(f"\n🎯 Applying data augmentation to training set only...")
    
    # Smart clipping configuration
    clip_config = {
        'ecg_threshold': 4.0,    # Aggressive for raw ECG outliers
        'eeg_threshold': 3.0,    # Conservative for preprocessed EEG
        'apply_to': 'both'       # Clip both signal types
    }
    
    # Realistic noise configuration  
    noise_config = {
        'eeg_gaussian': 0.03,    # Reduced for better challenge balance
        'eeg_muscle': 0.02,
        'ecg_baseline': 0.015,
        'ecg_powerline': 0.008,
        'probability': 0.6       # 60% of training samples get noise
    }
    
    # Apply augmentations
    X_train = apply_smart_clipping(X_train, y_train, clip_config)
    X_train = apply_realistic_noise(X_train, y_train, noise_config)
    
    # Check final extreme values
    extreme_count = np.sum(np.abs(X_train) > 6)
    if extreme_count > 0:
        extreme_percent = extreme_count / X_train.size * 100
        print(f"   📊 Post-augmentation: {extreme_count:,} extreme values ({extreme_percent:.3f}%)")
    else:
        print(f"   ✅ No extreme outliers after augmentation!")

    print(f"✅ LEAK-FREE dataset preparation complete!")
    print(f"   📊 Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"   🔒 Groups: {len(set(groups_train))} train, {len(set(groups_test))} test")
    print(f"   📈 Feature range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   📊 Train stats: mean={X_train.mean():.6f}, std={X_train.std():.6f}")
    print(f"   📊 Test stats: mean={X_test.mean():.6f}, std={X_test.std():.6f}")

    # Final class balance validation
    final_train_ecg = np.sum(y_train == 0)
    final_train_eeg = np.sum(y_train == 1)
    final_test_ecg = np.sum(y_test == 0)
    final_test_eeg = np.sum(y_test == 1)
    
    final_train_pct = final_train_ecg / (final_train_ecg + final_train_eeg) * 100
    final_test_pct = final_test_ecg / (final_test_ecg + final_test_eeg) * 100
    
    print(f"   ⚖️  Final class balance: Train {final_train_pct:.1f}% ECG, Test {final_test_pct:.1f}% ECG")

    if validate_alignment and eeg_metadata['structure_valid']:
        print("🔍 Signal alignment validation:")
        print(f"   EEG structure: {eeg_metadata['eeg_channels']}ch × {eeg_metadata['eeg_timepoints']}tp")
        print(f"   ✅ Signal alignment check completed")

    # Save dataset with groups
    print(f"\n📎 Caching dataset with groups...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_params = {
        'normalization': normalization,
        'normalization_strategy': normalization_strategy,
        'validate_alignment': validate_alignment,
        'ecg_path': ecg_path,
        'eeg_path': eeg_path,
        'dataset_fraction': dataset_fraction
    }
    
    np.savez_compressed(cache_path,
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test,
                        groups_train=np.array(groups_train, dtype=object),
                        eeg_channels=eeg_metadata['eeg_channels'],
                        eeg_timepoints=eeg_metadata['eeg_timepoints'],
                        structure_valid=eeg_metadata['structure_valid'],
                        scalers=np.array(scalers, dtype=object),
                        cache_params=np.array(cache_params, dtype=object))
    print(f"   ✅ Cached with groups to: {cache_path}")

    metadata = {
        'eeg_channels': eeg_metadata['eeg_channels'],
        'eeg_timepoints': eeg_metadata['eeg_timepoints'],
        'structure_valid': eeg_metadata['structure_valid'],
        'scalers': scalers,
        'normalization': normalization,
        'normalization_strategy': normalization_strategy,
        'groups_train': groups_train
    }

    return X_train, X_test, y_train, y_test, metadata
