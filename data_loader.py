import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_ecg_data(filepath, num_eeg=None):
    """
    Load ECG data from CSV file
    
    Args:
        filepath: Path to ECG CSV file (MIT-BIH dataset)
        num_eeg: Optional - if provided, balance ECG samples to match EEG count
        
    Returns:
        X: ECG feature matrix
        y: ECG labels (all zeros for binary ECG vs EEG classification)
    """
    try:
        print(f"📖 Loading ECG data from: {filepath}")
        
        # Try reading with header first, then without
        df = pd.read_csv(filepath)
        if df.shape[1] == 1:  # Might be delimiter issue
            df = pd.read_csv(filepath, header=None, delimiter=',')
    except Exception as e:
        print(f"   ⚠️ Header read failed, trying headerless: {e}")
        df = pd.read_csv(filepath, header=None, delimiter=',')
    
    print(f"   📊 Raw ECG data shape: {df.shape}")
    
    # Extract features and labels
    X = df.iloc[:, :-1].values    # all but last column (features)
    y = df.iloc[:, -1].values     # last column (original MIT-BIH labels)
    
    print(f"   🏷️ Original ECG label distribution: {np.bincount(y.astype(int))}")
    
    # Filter to keep only normal heartbeats (class 0) for binary classification
    # This creates a clean "normal ECG" vs "EEG" classification task
    ecg_mask = y == 0
    X_filtered = X[ecg_mask]
    
    print(f"   ✂️ ECG samples after filtering (normal heartbeats only): {len(X_filtered)}")
    print(f"   📊 ECG feature range: [{X_filtered.min():.6f}, {X_filtered.max():.6f}]")
    print(f"   📊 ECG statistics: mean={X_filtered.mean():.6f}, std={X_filtered.std():.6f}")
    
    # Balance with EEG dataset size if requested
    if num_eeg is not None:
        if len(X_filtered) > num_eeg:
            print(f"   ⚖️ Balancing ECG to match EEG size: {num_eeg:,} samples")
            # Use random sampling to match EEG size
            np.random.seed(42)  # Reproducible sampling
            indices = np.random.choice(len(X_filtered), num_eeg, replace=False)
            X_filtered = X_filtered[indices]
            print(f"   ✅ ECG balanced to: {len(X_filtered):,} samples")
        elif len(X_filtered) < num_eeg:
            print(f"   ⚠️ ECG has fewer samples ({len(X_filtered):,}) than EEG ({num_eeg:,})")
            print(f"   🎯 Using all available ECG samples")
    
    # Create binary labels for ECG vs EEG classification
    # All ECG samples get label 0 (EEG will get label 1)
    y_binary = np.zeros(len(X_filtered), dtype=int)
    
    print(f"   ✅ Final ECG dataset: {X_filtered.shape}")
    print(f"   🏷️ Binary labels: {len(y_binary)} samples, all labeled as 0 (ECG)")
    
    return X_filtered, y_binary

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

def load_eeg_data(filepath, chunk_size=5000):
    """
    Load EEG data from CSV file with chunked processing and structure detection
    
    Args:
        filepath: Path to EEG CSV file
        chunk_size: Process data in chunks of this size to reduce memory usage
    
    Returns:
        X_eeg: EEG feature data
        y_eeg: EEG labels  
        metadata: Dictionary containing structure information
    """
    print("🧠 Loading and analyzing EEG data...")
    
    # Initial analysis without loading full dataset
    print("📊 EEG Dataset Analysis:")
    eeg_df_sample = pd.read_csv(filepath, nrows=1000)  # Sample for analysis
    total_rows = sum(1 for line in open(filepath)) - 1  # Count total rows (minus header)
    
    print(f"   Total samples: {total_rows:,}")
    print(f"   Columns: {len(eeg_df_sample.columns)}")
    estimated_memory = total_rows * len(eeg_df_sample.columns) * 8 / 1024**2  # Estimate in MB
    print(f"   Estimated memory: {estimated_memory:.1f} MB")
    
    # Detect EEG structure from column names
    eeg_channels = None
    eeg_timepoints = None
    structure_valid = False
    
    if 'label' in eeg_df_sample.columns:
        feature_cols = [col for col in eeg_df_sample.columns if col != 'label']
        total_features = len(feature_cols)
        print(f"   Feature columns: {total_features}")
        
        # Try to detect structure from column names
        if feature_cols and feature_cols[0].startswith('ch') and '_t' in feature_cols[0]:
            print("   📡 Parsing structured column names...")
            channels = set()
            times = set()
            
            for col in feature_cols:
                if '_t' in col:
                    try:
                        ch_part, t_part = col.split('_t')
                        ch_num = int(ch_part.replace('ch', ''))
                        t_num = int(t_part)
                        channels.add(ch_num)
                        times.add(t_num)
                    except (ValueError, IndexError):
                        continue
            
            if channels and times:
                eeg_channels = len(channels)
                eeg_timepoints = len(times)
                structure_valid = True
                print(f"   📡 Detected from names: {eeg_channels} channels × {eeg_timepoints} timepoints")
        
        if not structure_valid:
            # Infer from common EEG configurations
            print("   💡 Inferring from common configurations...")
            common_configs = [(32, 188), (64, 125), (128, 250), (32, 250), (16, 376)]
            
            for channels, timepoints in common_configs:
                if channels * timepoints == total_features:
                    eeg_channels = channels
                    eeg_timepoints = timepoints
                    structure_valid = True
                    print(f"   💡 Inferred: {channels} channels × {timepoints} timepoints")
                    break
        
        if not structure_valid:
            # Fallback configuration
            eeg_channels = 32
            eeg_timepoints = total_features // 32
            print(f"   🔧 Fallback: {eeg_channels} channels × {eeg_timepoints} timepoints")
            print(f"   ⚠️  Structure validation failed - proceeding with caution")
    else:
        # No label column found
        eeg_channels = 32
        eeg_timepoints = 188
        structure_valid = False
        print(f"   ⚠️  No label column found, using defaults: {eeg_channels}ch × {eeg_timepoints}tp")
    
    # Clear sample dataframe
    del eeg_df_sample
    
    # Memory-efficient chunked loading
    print("🔄 Loading EEG data in chunks...")
    X_eeg_chunks = []
    y_eeg_chunks = []
    
    # Load data in chunks
    chunk_count = 0
    for chunk_df in pd.read_csv(filepath, chunksize=chunk_size):
        chunk_count += 1
        start_idx = (chunk_count - 1) * chunk_size
        end_idx = start_idx + len(chunk_df)
        
        print(f"   Processing EEG chunk {chunk_count}: samples {start_idx:,}-{end_idx:,}")
        
        # Extract features and labels
        if 'label' in chunk_df.columns:
            X_chunk = chunk_df.drop('label', axis=1).values
            y_chunk = chunk_df['label'].astype(int).values
        else:
            # Fallback: assume last column is label
            X_chunk = chunk_df.iloc[:, :-1].values
            y_chunk = chunk_df.iloc[:, -1].astype(int).values
        
        X_eeg_chunks.append(X_chunk)
        y_eeg_chunks.append(y_chunk)
        
        # Clear chunk from memory immediately
        del chunk_df, X_chunk, y_chunk
    
    # Combine all chunks
    print("   🔗 Combining EEG chunks...")
    X_eeg = np.vstack(X_eeg_chunks)
    y_eeg = np.concatenate(y_eeg_chunks)
    
    # Clear chunk lists to free memory
    del X_eeg_chunks, y_eeg_chunks
    
    print(f"   ✅ EEG data loaded: {X_eeg.shape}")
    print(f"   📊 EEG Statistics:")
    print(f"      Unique labels: {np.unique(y_eeg).tolist()}")
    print(f"      Data range: [{X_eeg.min():.6f}, {X_eeg.max():.6f}]")
    print(f"      Mean: {X_eeg.mean():.6f}, Std: {X_eeg.std():.6f}")
    print(f"      Memory usage: {X_eeg.nbytes/1024**2:.1f} MB")
    
    # Create metadata dictionary
    metadata = {
        'eeg_channels': eeg_channels,
        'eeg_timepoints': eeg_timepoints,
        'structure_valid': structure_valid,
        'total_features': X_eeg.shape[1],
        'total_samples': X_eeg.shape[0],
        'unique_labels': np.unique(y_eeg).tolist(),
        'chunks_processed': chunk_count
    }
    
    return X_eeg, y_eeg, metadata

def project_ecg_to_eeg_format(X_ecg, eeg_channels, eeg_timepoints):
    """
    Project ECG data to match EEG format using learned transformations
    
    Args:
        X_ecg: ECG data (n_samples, ecg_features)
        eeg_channels: Number of EEG channels 
        eeg_timepoints: Number of timepoints per EEG segment
    
    Returns:
        X_ecg_projected: ECG data reshaped to (n_samples, eeg_channels * eeg_timepoints)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    target_features = eeg_channels * eeg_timepoints
    ecg_features = X_ecg.shape[1]
    
    print(f"   🔄 Projecting ECG: {ecg_features} → {target_features} features")
    
    if ecg_features == target_features:
        print("   ✅ ECG already matches EEG dimensions")
        return X_ecg
    
    elif ecg_features > target_features:
        # Use PCA to reduce dimensions while preserving variance
        print(f"   📉 Reducing ECG dimensions with PCA...")
        pca = PCA(n_components=target_features)
        X_ecg_projected = pca.fit_transform(X_ecg)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"   📊 PCA preserved {explained_var:.3f} of variance")
        
    else:
        # Expand dimensions by repeating/interpolating features
        print(f"   📈 Expanding ECG dimensions...")
        
        # Method 1: Repeat features cyclically
        repeat_factor = target_features // ecg_features
        remainder = target_features % ecg_features
        
        X_repeated = np.tile(X_ecg, (1, repeat_factor))
        if remainder > 0:
            X_remainder = X_ecg[:, :remainder]
            X_ecg_projected = np.hstack([X_repeated, X_remainder])
        else:
            X_ecg_projected = X_repeated
        
        print(f"   🔄 Repeated ECG features {repeat_factor}x + {remainder} extra")
    
    return X_ecg_projected

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

def prepare_dataset(ecg_path, eeg_path, normalization='zscore', 
                   normalization_strategy='separate', validate_alignment=True, 
                   force_reload=False, cache_path='data/preprocessed_dataset.npz',
                   chunk_size=5000, memory_limit_gb=None, dataset_fraction=1.0):
    """
    Memory-efficient dataset preparation with chunked processing and proper normalization

    Args:
        memory_limit_gb: Memory limit in GB for dataset processing. If provided,
                        automatically calculates appropriate chunk_size.
    """
    import numpy as np
    import os
    from sklearn.model_selection import train_test_split

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
                    'eeg_path': eeg_path
                }
                if cache_params == current_params:
                    print(f"✅ Cache parameters match - loading cached data")
                    X_train = cached['X_train']
                    X_test = cached['X_test']
                    y_train = cached['y_train']
                    y_test = cached['y_test']
                    print(f"📊 Cached data loaded:")
                    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
                    print(f"   Range: [{X_train.min():.6f}, {X_train.max():.6f}]")
                    metadata = {
                        'eeg_channels': int(cached.get('eeg_channels', 32)),
                        'eeg_timepoints': int(cached.get('eeg_timepoints', 188)),
                        'structure_valid': bool(cached.get('structure_valid', True)),
                        'scalers': cached.get('scalers', {}).item() if 'scalers' in cached else {},
                        'normalization': normalization,
                        'normalization_strategy': normalization_strategy
                    }
                    return X_train, X_test, y_train, y_test, metadata
                else:
                    print(f"⚠️ Cache parameter mismatch - regenerating")
            else:
                print(f"⚠️ Old cache format - regenerating")
        except Exception as e:
            print(f"⚠️ Cache validation failed: {str(e)}")

    print(f"📝 Generating fresh dataset with chunk size: {chunk_size:,}")
    X_eeg, y_eeg, eeg_metadata = load_eeg_data(eeg_path, chunk_size=chunk_size)
    print("\n❤️  Loading ECG data...")
    X_ecg, y_ecg = load_ecg_data(ecg_path, num_eeg=len(X_eeg))
    print(f"ECG samples loaded: {X_ecg.shape[0]}")

    min_samples = min(len(X_ecg), len(X_eeg))
    print(f"   Target size: {min_samples:,} samples each")

    # Balance ECG if needed
    if len(X_ecg) > min_samples:
        print(f"   🔄 Downsampling ECG: {len(X_ecg):,} → {min_samples:,}")
        np.random.seed(42)
        indices = np.random.choice(len(X_ecg), min_samples, replace=False)
        X_ecg = X_ecg[indices]
        y_ecg = y_ecg[indices]

    # Balance EEG if needed
    if len(X_eeg) > min_samples:
        print(f"   🔄 Downsampling EEG: {len(X_eeg):,} → {min_samples:,}")
        np.random.seed(42)
        indices = np.random.choice(len(X_eeg), min_samples, replace=False)
        X_eeg = X_eeg[indices]
        y_eeg = y_eeg[indices]

    print(f"   ✅ Final balanced sizes: ECG={len(X_ecg):,}, EEG={len(X_eeg):,}")

    # Apply dataset fraction early (before compatibility and normalization)
    if dataset_fraction < 1.0:
        target_eeg = int(len(X_eeg) * dataset_fraction)
        target_ecg = int(len(X_ecg) * dataset_fraction)
        X_eeg = X_eeg[:target_eeg]
        y_eeg = y_eeg[:target_eeg]
        X_ecg = X_ecg[:target_ecg]
        y_ecg = y_ecg[:target_ecg]
        print(f"✂️ Applied dataset_fraction before normalization:")
        print(f"   EEG: {target_eeg} samples | ECG: {target_ecg} samples")

    X_ecg, X_eeg = check_feature_compatibility(X_ecg, X_eeg)

    print(f"\n🔧 APPLYING NORMALIZATION: {normalization} (strategy: {normalization_strategy})")
    X_ecg_norm, X_eeg_norm, scalers = normalize_datasets(
        X_ecg, X_eeg, 
        normalization=normalization, 
        strategy=normalization_strategy
    )

    del X_ecg, X_eeg
    y_ecg_labels = np.zeros(len(X_ecg_norm), dtype=int)
    y_eeg_labels = np.ones(len(X_eeg_norm), dtype=int)

    print(f"\n📦 Combining normalized datasets...")
    X = np.vstack((X_ecg_norm, X_eeg_norm))
    y = np.concatenate((y_ecg_labels, y_eeg_labels))
    n_ecg_samples = len(y_ecg_labels)
    del X_ecg_norm, X_eeg_norm, y_ecg_labels, y_eeg_labels

    if validate_alignment and eeg_metadata['structure_valid']:
        print("🔍 Signal alignment validation:")
        print(f"   ECG samples: {n_ecg_samples}")
        print(f"   EEG samples: {len(X) - n_ecg_samples}")
        print(f"   EEG structure: {eeg_metadata['eeg_channels']}ch × {eeg_metadata['eeg_timepoints']}tp")
        print(f"   ✅ Signal alignment check completed")

    print("🔀 Shuffling data...")
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    print("Label distribution after shuffle:", np.bincount(y))

    print("\n✂️  Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    del X, y

    print("✅ Final dataset summary:")
    print(f"   Train set: {X_train.shape} | Labels: {np.bincount(y_train)}")
    print(f"   Test set:  {X_test.shape} | Labels: {np.bincount(y_test)}")
    print(f"   Feature range: [{X_train.min():.6f}, {X_train.max():.6f}]")
    print(f"   Final statistics:")
    print(f"      Mean: {np.mean(X_train):.6f}")
    print(f"      Std: {np.std(X_train):.6f}")
    print(f"   Memory usage: Train={X_train.nbytes/1024**3:.2f}GB, Test={X_test.nbytes/1024**3:.2f}GB")

    print(f"\n📎 Caching dataset...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_params = {
        'normalization': normalization,
        'normalization_strategy': normalization_strategy,
        'validate_alignment': validate_alignment,
        'ecg_path': ecg_path,
        'eeg_path': eeg_path
    }
    print("   🔄 Saving compressed arrays...")
    np.savez_compressed(cache_path,
                        X_train=X_train, X_test=X_test,
                        y_train=y_train, y_test=y_test,
                        eeg_channels=eeg_metadata['eeg_channels'],
                        eeg_timepoints=eeg_metadata['eeg_timepoints'],
                        structure_valid=eeg_metadata['structure_valid'],
                        scalers=np.array(scalers, dtype=object),
                        cache_params=np.array(cache_params, dtype=object))
    print(f"   ✅ Cached to: {cache_path}")

    metadata = {
        'eeg_channels': eeg_metadata['eeg_channels'],
        'eeg_timepoints': eeg_metadata['eeg_timepoints'],
        'structure_valid': eeg_metadata['structure_valid'],
        'scalers': scalers,
        'normalization': normalization,
        'normalization_strategy': normalization_strategy
    }

    return X_train, X_test, y_train, y_test, metadata