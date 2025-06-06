import pandas as pd
import numpy as np
import os

def validate_eeg_csv(csv_path):
    """
    Validate the structure and validity of the generated EEG CSV dataset.
    """
    print(f"\n🔍 Validating EEG dataset: {csv_path}\n")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return False
    
    try:
        # Load the dataset with error handling
        print("📊 Loading dataset...")
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.ParserError as e:
            print(f"⚠️  CSV parsing error detected: {e}")
            print("🔧 Attempting to load with error recovery...")
            
            # Try to load with error recovery
            df = pd.read_csv(csv_path, error_bad_lines=False, warn_bad_lines=True)
            print(f"✅ Loaded with error recovery - some malformed rows were skipped")
        
        # Basic shape information
        n_rows, n_cols = df.shape
        print(f"✅ Dataset loaded successfully")
        print(f"📐 Shape: {n_rows} rows × {n_cols} columns")
        
        # Expected calculations
        SEGMENT_LENGTH = 188
        DOWNSAMPLED_FREQ = 128
        TRIAL_DURATION = 63.0
        N_TRIALS_DEAP = 40
        N_FILES_DEAP = 32  # DEAP has 32 participants (s01.bdf to s32.bdf)
        EXPECTED_EEG_CHANNELS = 32  # DEAP standard
        
        # Calculate expected dimensions
        samples_per_trial = int(TRIAL_DURATION * DOWNSAMPLED_FREQ)  # 63 * 128 = 8064
        segments_per_trial = samples_per_trial // SEGMENT_LENGTH  # 8064 // 188 = 42 (with remainder)
        
        print(f"\n📋 EXPECTED STRUCTURE ANALYSIS:")
        print(f"   ⏱️  Trial duration: {TRIAL_DURATION}s")
        print(f"   🔄 Sampling frequency: {DOWNSAMPLED_FREQ} Hz") 
        print(f"   📏 Samples per trial: {samples_per_trial}")
        print(f"   🧩 Segment length: {SEGMENT_LENGTH} samples")
        print(f"   📦 Max segments per trial: {segments_per_trial}")
        print(f"   🎯 Trials per file: {N_TRIALS_DEAP}")
        
        # Determine number of EEG channels from columns
        label_cols = [col for col in df.columns if col.lower() == 'label']
        eeg_cols = [col for col in df.columns if col not in label_cols]
        n_eeg_features = len(eeg_cols)
        n_channels = n_eeg_features // SEGMENT_LENGTH
        
        print(f"\n🧠 CHANNEL ANALYSIS:")
        print(f"   📡 EEG feature columns: {n_eeg_features}")
        print(f"   🎛️  Detected EEG channels: {n_channels} (expected: {EXPECTED_EEG_CHANNELS})")
        print(f"   🏷️  Label columns: {len(label_cols)}")
        print(f"   ✅ Expected features per segment: {EXPECTED_EEG_CHANNELS} × {SEGMENT_LENGTH} = {EXPECTED_EEG_CHANNELS * SEGMENT_LENGTH}")
        
        # Check if channel count matches expectation
        if n_channels != EXPECTED_EEG_CHANNELS:
            print(f"⚠️  Channel count mismatch: got {n_channels}, expected {EXPECTED_EEG_CHANNELS}")
        else:
            print(f"✅ Channel count correct: {n_channels}")
        
        # Validate column structure
        expected_total_cols = EXPECTED_EEG_CHANNELS * SEGMENT_LENGTH + 1  # +1 for label
        if n_cols != expected_total_cols:
            print(f"⚠️  Column count mismatch: got {n_cols}, expected {expected_total_cols}")
            print(f"   📊 Difference: {n_cols - expected_total_cols} columns")
        else:
            print(f"✅ Column count correct: {n_cols}")
        
        # Estimate expected rows
        expected_segments_per_file = N_TRIALS_DEAP * segments_per_trial
        expected_total_segments = N_FILES_DEAP * expected_segments_per_file
        
        print(f"\n📊 ROW COUNT ANALYSIS:")
        print(f"   📦 Max segments per file: {expected_segments_per_file}")
        print(f"   📁 Expected total segments (32 files): ~{expected_total_segments}")
        print(f"   📈 Actual segments: {n_rows}")
        
        if n_rows < expected_total_segments * 0.8:  # Allow 20% variance
            print(f"⚠️  Fewer segments than expected (possible short files or errors)")
        elif n_rows > expected_total_segments * 1.2:
            print(f"⚠️  More segments than expected (check for duplicates or overlap)")
        else:
            print(f"✅ Segment count within expected range")
        
        # Data type validation
        print(f"\n🔢 DATA TYPE ANALYSIS:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        
        print(f"   🔢 Numeric columns: {len(numeric_cols)}")
        print(f"   ❓ Non-numeric columns: {len(non_numeric_cols)}")
        
        if len(non_numeric_cols) > 0:
            print(f"   ⚠️  Non-numeric columns found: {list(non_numeric_cols)}")
        else:
            print(f"   ✅ All columns are numeric")
        
        # Check for missing values
        print(f"\n🕳️  MISSING VALUES CHECK:")
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"   ❌ Missing values found: {missing_count}")
            cols_with_missing = df.columns[df.isnull().any()].tolist()
            print(f"   📋 Columns with missing values: {cols_with_missing}")
        else:
            print(f"   ✅ No missing values found")
        
        # Label analysis
        print(f"\n🏷️  LABEL ANALYSIS:")
        if 'label' in df.columns:
            unique_labels = df['label'].unique()
            label_counts = df['label'].value_counts()
            print(f"   🎯 Unique labels: {unique_labels}")
            print(f"   📊 Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                print(f"      Label {label}: {count} ({percentage:.1f}%)")
        else:
            print(f"   ❌ No 'label' column found")
        
        # EEG data range analysis (sample a few columns)
        print(f"\n⚡ EEG DATA RANGE ANALYSIS:")
        sample_eeg_cols = eeg_cols[:5]  # Sample first 5 EEG columns
        for col in sample_eeg_cols:
            col_data = df[col]
            print(f"   📡 {col}: min={col_data.min():.3f}, max={col_data.max():.3f}, mean={col_data.mean():.3f}")
        
        # Check for suspicious values
        print(f"\n🚨 ANOMALY CHECK:")
        
        # Check for identical rows (potential duplicates)
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            print(f"   ⚠️  Duplicate rows found: {duplicate_count}")
        else:
            print(f"   ✅ No duplicate rows")
        
        # Check for constant columns
        constant_cols = []
        for col in eeg_cols[:10]:  # Check first 10 EEG columns
            if df[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"   ⚠️  Constant value columns: {constant_cols}")
        else:
            print(f"   ✅ No constant value columns (in sample)")
        
        # Memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"\n💾 MEMORY USAGE:")
        print(f"   📏 Dataset size: {memory_usage_mb:.1f} MB")
        
        # File size
        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"   💿 File size on disk: {file_size_mb:.1f} MB")
        
        print(f"\n✅ VALIDATION COMPLETE")
        print(f"📋 Summary: {n_rows} segments × {n_channels} channels × {SEGMENT_LENGTH} timepoints + 1 label")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False

def quick_sample_check(csv_path, n_samples=5):
    """
    Display a few sample rows for manual inspection
    """
    print(f"\n👀 SAMPLE DATA PREVIEW:")
    try:
        df = pd.read_csv(csv_path, nrows=n_samples)
        
        # Show first few and last few columns
        print(f"   First 5 columns:")
        print(df.iloc[:, :5].to_string(index=False))
        
        print(f"\n   Last 2 columns (including label):")
        print(df.iloc[:, -2:].to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error showing sample: {e}")

if __name__ == "__main__":
    # Update this path to your actual CSV file
    CSV_PATH = "data/eeg_dataset_32.csv"
    
    # Run validation
    success = validate_eeg_csv(CSV_PATH)
    
    if success:
        # Show sample data
        quick_sample_check(CSV_PATH)
    
    print(f"\n🎉 Validation script completed!")