#!/usr/bin/env python3
"""
Standalone analysis of ECG and EEG datasets to understand their structure and preprocessing
"""

import pandas as pd
import numpy as np
import os

def analyze_dataset_structure(filepath, dataset_name, sample_rows=1000):
    """Comprehensive analysis of dataset structure and statistics"""
    
    print(f"\n{'='*60}")
    print(f"🔍 ANALYZING {dataset_name.upper()} DATASET")
    print(f"📁 File: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        # Get file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"💾 File size: {file_size_mb:.1f} MB")
        
        # Read header info first
        print(f"\n📋 DATASET STRUCTURE:")
        df_head = pd.read_csv(filepath, nrows=5)
        print(f"   Shape (first 5 rows): {df_head.shape}")
        print(f"   Columns: {list(df_head.columns[:10])}{'...' if len(df_head.columns) > 10 else ''}")
        
        # Get full dataset info (without loading all data)
        print(f"\n📊 FULL DATASET INFO:")
        # Count total rows efficiently
        with open(filepath, 'r') as f:
            total_rows = sum(1 for line in f) - 1  # -1 for header
        total_columns = len(df_head.columns)
        
        print(f"   Total shape: ({total_rows:,}, {total_columns:,})")
        print(f"   Total features (excluding label): {total_columns - 1:,}")
        print(f"   Estimated memory if loaded: {(total_rows * total_columns * 8) / (1024**3):.2f} GB")
        
        # Analyze column structure
        print(f"\n🏷️  COLUMN ANALYSIS:")
        feature_cols = [col for col in df_head.columns if col != 'label']
        
        if 'label' in df_head.columns:
            print(f"   Label column: Found ('label')")
            label_values = df_head['label'].unique()
            print(f"   Label values in sample: {label_values}")
        else:
            print(f"   Label column: Not found - assuming last column")
            print(f"   Last column name: '{df_head.columns[-1]}'")
            label_values = df_head.iloc[:, -1].unique()
            print(f"   Last column values: {label_values}")
        
        # Analyze feature naming pattern
        print(f"   Feature columns: {len(feature_cols)}")
        print(f"   First 10 features: {feature_cols[:10]}")
        print(f"   Last 10 features: {feature_cols[-10:]}")
        
        # Check if it follows channel×timepoint structure
        if any('ch' in col and '_t' in col for col in feature_cols):
            print(f"   ✅ Detected channel×timepoint structure (ch*_t*)")
            
            # Extract channel and timepoint info
            channels = set()
            timepoints = set()
            for col in feature_cols:
                if 'ch' in col and '_t' in col:
                    try:
                        parts = col.split('_')
                        ch_part = [p for p in parts if p.startswith('ch')][0]
                        t_part = [p for p in parts if p.startswith('t')][0]
                        
                        ch_num = int(ch_part.replace('ch', ''))
                        t_num = int(t_part.replace('t', ''))
                        
                        channels.add(ch_num)
                        timepoints.add(t_num)
                    except:
                        pass
            
            if channels and timepoints:
                print(f"   📊 Detected structure:")
                print(f"      Channels: {min(channels)} to {max(channels)} ({len(channels)} total)")
                print(f"      Timepoints: {min(timepoints)} to {max(timepoints)} ({len(timepoints)} total)")
                print(f"      Expected features: {len(channels)} × {len(timepoints)} = {len(channels) * len(timepoints)}")
                print(f"      Actual features: {len(feature_cols)}")
                
                if len(channels) * len(timepoints) == len(feature_cols):
                    print(f"      ✅ Structure is complete and consistent")
                else:
                    print(f"      ⚠️ Structure mismatch!")
        
        # Sample data for statistical analysis
        print(f"\n📈 STATISTICAL ANALYSIS (sample of {sample_rows:,} rows):")
        df_sample = pd.read_csv(filepath, nrows=sample_rows)
        
        # Remove label column for feature analysis
        if 'label' in df_sample.columns:
            X_sample = df_sample.drop('label', axis=1)
            y_sample = df_sample['label']
        else:
            X_sample = df_sample.iloc[:, :-1]
            y_sample = df_sample.iloc[:, -1]
        
        print(f"   Sample shape: {X_sample.shape}")
        print(f"   Data type: {X_sample.dtypes.iloc[0]}")
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        overall_mean = np.mean(X_sample.values)
        overall_std = np.std(X_sample.values)
        overall_min = np.min(X_sample.values)
        overall_max = np.max(X_sample.values)
        
        print(f"   Mean: {overall_mean:.6f}")
        print(f"   Std: {overall_std:.6f}")
        print(f"   Range: [{overall_min:.6f}, {overall_max:.6f}]")
        
        # Check for signs of preprocessing
        print(f"\n🔍 PREPROCESSING DETECTION:")
        
        # Check for normalization patterns
        is_likely_minmax = (overall_min >= -0.1 and overall_max <= 1.1 and overall_max > 0.8)
        is_likely_zscore = (abs(overall_mean) < 0.1 and abs(overall_std - 1.0) < 0.2)
        is_likely_raw = (overall_max > 10 or overall_min < -10)
        
        if is_likely_minmax:
            print(f"   🎯 LIKELY MIN-MAX NORMALIZED (range ≈ [0,1])")
        elif is_likely_zscore:
            print(f"   🎯 LIKELY Z-SCORE NORMALIZED (mean≈0, std≈1)")
        elif is_likely_raw:
            print(f"   🎯 LIKELY RAW/UNNORMALIZED DATA")
        else:
            print(f"   🤷 UNCLEAR PREPROCESSING STATE")
        
        # Check for problematic values
        nan_count = np.sum(np.isnan(X_sample.values))
        inf_count = np.sum(np.isinf(X_sample.values))
        zero_count = np.sum(X_sample.values == 0)
        
        print(f"   NaN values: {nan_count:,}")
        print(f"   Infinite values: {inf_count:,}")
        print(f"   Zero values: {zero_count:,} ({zero_count/X_sample.size*100:.1f}%)")
        
        # Percentile analysis
        print(f"\n📊 PERCENTILE DISTRIBUTION:")
        percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
        values = np.percentile(X_sample.values.flatten(), percentiles)
        
        for p, v in zip(percentiles, values):
            print(f"     {p:5.1f}%: {v:12.6f}")
        
        # Label analysis
        print(f"\n🏷️  LABEL ANALYSIS:")
        label_counts = y_sample.value_counts().sort_index()
        print(f"   Unique labels: {list(label_counts.index)}")
        print(f"   Label distribution:")
        for label, count in label_counts.items():
            percentage = count / len(y_sample) * 100
            print(f"     Label {label}: {count:,} samples ({percentage:.1f}%)")
        
        # Feature variance analysis (first 20 features)
        print(f"\n📊 FEATURE VARIANCE ANALYSIS (first 20 features):")
        for i in range(min(20, X_sample.shape[1])):
            feature_data = X_sample.iloc[:, i]
            f_mean = np.mean(feature_data)
            f_std = np.std(feature_data)
            f_min, f_max = np.min(feature_data), np.max(feature_data)
            f_var = np.var(feature_data)
            
            print(f"   {X_sample.columns[i]}: mean={f_mean:8.4f}, std={f_std:8.4f}, var={f_var:8.4f}, range=[{f_min:8.4f}, {f_max:8.4f}]")
        
        # Check for consistent patterns across features
        print(f"\n🔍 FEATURE CONSISTENCY CHECK:")
        feature_means = X_sample.mean(axis=0)
        feature_stds = X_sample.std(axis=0)
        
        mean_of_means = np.mean(feature_means)
        std_of_means = np.std(feature_means)
        mean_of_stds = np.mean(feature_stds)
        std_of_stds = np.std(feature_stds)
        
        print(f"   Feature means: avg={mean_of_means:.6f}, std={std_of_means:.6f}")
        print(f"   Feature stds:  avg={mean_of_stds:.6f}, std={std_of_stds:.6f}")
        
        if std_of_means < 0.1 and std_of_stds < 0.1:
            print(f"   ✅ Features are very consistent (likely normalized)")
        elif std_of_means > 1.0 or std_of_stds > 1.0:
            print(f"   ⚠️ Features vary significantly (likely raw/unnormalized)")
        
        print(f"\n{'='*60}")
        print(f"✅ Analysis complete for {dataset_name.upper()}")
        print(f"{'='*60}")
        
        return {
            'shape': (total_rows, total_columns),
            'file_size_mb': file_size_mb,
            'overall_stats': {
                'mean': overall_mean,
                'std': overall_std,
                'min': overall_min,
                'max': overall_max
            },
            'preprocessing_likely': {
                'minmax': is_likely_minmax,
                'zscore': is_likely_zscore,
                'raw': is_likely_raw
            },
            'label_distribution': dict(label_counts),
            'has_issues': nan_count > 0 or inf_count > 0
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main analysis function"""
    
    print("🔍 STANDALONE DATASET ANALYSIS")
    print("=" * 80)
    
    # Define dataset paths
    ecg_path = "data/mitbih_train.csv"  # Adjust if needed
    eeg_path = "data/eeg_dataset_32.csv"
    
    results = {}
    
    # Analyze ECG dataset
    if os.path.exists(ecg_path):
        results['ecg'] = analyze_dataset_structure(ecg_path, "ECG")
    else:
        print(f"\n❌ ECG dataset not found: {ecg_path}")
        print("   Available CSV files in data/:")
        if os.path.exists("data/"):
            csv_files = [f for f in os.listdir("data/") if f.endswith('.csv')]
            for f in csv_files:
                print(f"     {f}")
    
    # Analyze EEG dataset  
    if os.path.exists(eeg_path):
        results['eeg'] = analyze_dataset_structure(eeg_path, "EEG")
    else:
        print(f"\n❌ EEG dataset not found: {eeg_path}")
        print("   Available CSV files in data/:")
        if os.path.exists("data/"):
            csv_files = [f for f in os.listdir("data/") if f.endswith('.csv')]
            for f in csv_files:
                print(f"     {f}")
    
    # Comparison summary
    if len(results) >= 2:
        print(f"\n" + "="*80)
        print(f"📊 COMPARISON SUMMARY")
        print(f"="*80)
        
        for dataset_name, result in results.items():
            if result:
                print(f"\n{dataset_name.upper()}:")
                print(f"   Shape: {result['shape']}")
                print(f"   Size: {result['file_size_mb']:.1f} MB")
                print(f"   Range: [{result['overall_stats']['min']:.6f}, {result['overall_stats']['max']:.6f}]")
                print(f"   Mean: {result['overall_stats']['mean']:.6f}")
                print(f"   Std: {result['overall_stats']['std']:.6f}")
                
                preprocessing = []
                if result['preprocessing_likely']['minmax']:
                    preprocessing.append("MIN-MAX NORMALIZED")
                if result['preprocessing_likely']['zscore']:
                    preprocessing.append("Z-SCORE NORMALIZED")
                if result['preprocessing_likely']['raw']:
                    preprocessing.append("RAW/UNNORMALIZED")
                
                print(f"   Likely preprocessing: {', '.join(preprocessing) if preprocessing else 'UNCLEAR'}")
                print(f"   Has issues: {'YES' if result['has_issues'] else 'NO'}")
        
        print(f"\n🎯 KEY INSIGHTS:")
        
        # Check if datasets are differently preprocessed
        if 'ecg' in results and 'eeg' in results:
            ecg_minmax = results['ecg']['preprocessing_likely']['minmax']
            eeg_minmax = results['eeg']['preprocessing_likely']['minmax']
            ecg_raw = results['ecg']['preprocessing_likely']['raw']
            eeg_raw = results['eeg']['preprocessing_likely']['raw']
            
            if ecg_minmax != eeg_minmax or ecg_raw != eeg_raw:
                print(f"   ⚠️ DATASETS HAVE DIFFERENT PREPROCESSING STATES!")
                print(f"      This explains the normalization issues in your pipeline.")
                print(f"      Recommendation: Normalize both datasets consistently before combining.")
            else:
                print(f"   ✅ Both datasets appear to have similar preprocessing states.")
    
    print(f"\n" + "="*80)
    print(f"✅ ANALYSIS COMPLETE")
    print(f"="*80)

if __name__ == "__main__":
    main()