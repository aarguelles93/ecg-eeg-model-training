#!/usr/bin/env python3
"""
Dataset Validation Script for ECG vs EEG Classification
Validates preprocessed_dataset.npz for shape, content, and quality issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

class DatasetValidator:
    def __init__(self, dataset_path='data/preprocessed_dataset.npz'):
        self.dataset_path = dataset_path
        self.data = None
        self.issues = []
        self.warnings = []
        self.stats = {}
        
    def load_dataset(self):
        """Load and inspect the preprocessed dataset"""
        print("🔍 DATASET VALIDATION REPORT")
        print("=" * 60)
        
        if not os.path.exists(self.dataset_path):
            self.issues.append(f"❌ Dataset file not found: {self.dataset_path}")
            return False
        
        try:
            self.data = np.load(self.dataset_path)
            print(f"✅ Successfully loaded: {self.dataset_path}")
            print(f"📁 File size: {os.path.getsize(self.dataset_path) / 1024**2:.1f} MB")
            return True
        except Exception as e:
            self.issues.append(f"❌ Failed to load dataset: {str(e)}")
            return False
    
    def validate_structure(self):
        """Validate dataset structure and keys"""
        print(f"\n📋 STRUCTURE VALIDATION")
        print("-" * 30)
        
        expected_keys = ['X_train', 'X_test', 'y_train', 'y_test']
        available_keys = list(self.data.keys())
        
        print(f"Available keys: {available_keys}")
        
        for key in expected_keys:
            if key in available_keys:
                print(f"✅ {key}: Found")
            else:
                self.issues.append(f"❌ Missing required key: {key}")
        
        # Check for extra keys
        extra_keys = set(available_keys) - set(expected_keys)
        if extra_keys:
            print(f"📋 Additional keys: {list(extra_keys)}")
        
        return len(self.issues) == 0
    
    def validate_shapes(self):
        """Validate data shapes and dimensions"""
        print(f"\n📏 SHAPE VALIDATION")
        print("-" * 30)
        
        X_train = self.data['X_train']
        X_test = self.data['X_test']
        y_train = self.data['y_train']
        y_test = self.data['y_test']
        
        # Basic shape info
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape:  {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape:  {y_test.shape}")
        
        # Validate consistency
        if X_train.shape[0] != y_train.shape[0]:
            self.issues.append("❌ X_train and y_train have different number of samples")
        else:
            print(f"✅ Training set: {X_train.shape[0]:,} samples")
        
        if X_test.shape[0] != y_test.shape[0]:
            self.issues.append("❌ X_test and y_test have different number of samples")
        else:
            print(f"✅ Test set: {X_test.shape[0]:,} samples")
        
        if X_train.shape[1] != X_test.shape[1]:
            self.issues.append("❌ Training and test sets have different feature dimensions")
        else:
            print(f"✅ Feature dimensions: {X_train.shape[1]:,}")
        
        # Store stats
        self.stats['total_samples'] = X_train.shape[0] + X_test.shape[0]
        self.stats['features'] = X_train.shape[1]
        self.stats['train_samples'] = X_train.shape[0]
        self.stats['test_samples'] = X_test.shape[0]
        
        # Memory usage
        total_memory = (X_train.nbytes + X_test.nbytes + y_train.nbytes + y_test.nbytes) / 1024**3
        print(f"💾 Total memory usage: {total_memory:.2f} GB")
        
        # Expected structure check
        expected_features = 32 * 188  # 32 channels × 188 timepoints = 6016
        if X_train.shape[1] == expected_features:
            print(f"✅ Feature count matches expected EEG structure (32 × 188 = {expected_features})")
        else:
            self.warnings.append(f"⚠️  Feature count ({X_train.shape[1]}) doesn't match expected EEG structure ({expected_features})")
    
    def validate_labels(self):
        """Validate label distribution and format"""
        print(f"\n🏷️  LABEL VALIDATION")
        print("-" * 30)
        
        y_train = self.data['y_train']
        y_test = self.data['y_test']
        
        # Check label types and ranges
        train_labels = np.unique(y_train)
        test_labels = np.unique(y_test)
        
        print(f"Training labels: {train_labels}")
        print(f"Test labels: {test_labels}")
        
        # Validate binary classification
        expected_labels = [0, 1]
        if not np.array_equal(sorted(train_labels), expected_labels):
            self.issues.append(f"❌ Training labels not binary: expected {expected_labels}, got {train_labels}")
        
        if not np.array_equal(sorted(test_labels), expected_labels):
            self.issues.append(f"❌ Test labels not binary: expected {expected_labels}, got {test_labels}")
        
        # Check label distribution
        train_counts = Counter(y_train)
        test_counts = Counter(y_test)
        
        print(f"\n📊 Training set distribution:")
        for label, count in train_counts.items():
            percentage = (count / len(y_train)) * 100
            label_name = "ECG" if label == 0 else "EEG"
            print(f"   {label_name} (label {label}): {count:,} samples ({percentage:.1f}%)")
        
        print(f"\n📊 Test set distribution:")
        for label, count in test_counts.items():
            percentage = (count / len(y_test)) * 100
            label_name = "ECG" if label == 0 else "EEG"
            print(f"   {label_name} (label {label}): {count:,} samples ({percentage:.1f}%)")
        
        # Check for severe class imbalance
        train_ratio = min(train_counts.values()) / max(train_counts.values())
        test_ratio = min(test_counts.values()) / max(test_counts.values())
        
        if train_ratio < 0.3:
            self.warnings.append(f"⚠️  Severe class imbalance in training set (ratio: {train_ratio:.2f})")
        
        if test_ratio < 0.3:
            self.warnings.append(f"⚠️  Severe class imbalance in test set (ratio: {test_ratio:.2f})")
        
        # Store stats
        self.stats['label_distribution'] = {
            'train': dict(train_counts),
            'test': dict(test_counts)
        }
    
    def validate_data_quality(self):
        """Check for data quality issues"""
        print(f"\n🔍 DATA QUALITY VALIDATION")
        print("-" * 30)
        
        X_train = self.data['X_train']
        X_test = self.data['X_test']
        
        # Check for NaN values
        train_nans = np.isnan(X_train).sum()
        test_nans = np.isnan(X_test).sum()
        
        if train_nans > 0:
            self.issues.append(f"❌ Training data contains {train_nans:,} NaN values")
        else:
            print("✅ No NaN values in training data")
        
        if test_nans > 0:
            self.issues.append(f"❌ Test data contains {test_nans:,} NaN values")
        else:
            print("✅ No NaN values in test data")
        
        # Check for infinite values
        train_infs = np.isinf(X_train).sum()
        test_infs = np.isinf(X_test).sum()
        
        if train_infs > 0:
            self.issues.append(f"❌ Training data contains {train_infs:,} infinite values")
        else:
            print("✅ No infinite values in training data")
        
        if test_infs > 0:
            self.issues.append(f"❌ Test data contains {test_infs:,} infinite values")
        else:
            print("✅ No infinite values in test data")
        
        # Check data ranges
        train_min, train_max = X_train.min(), X_train.max()
        test_min, test_max = X_test.min(), X_test.max()
        
        print(f"📊 Training data range: [{train_min:.6f}, {train_max:.6f}]")
        print(f"📊 Test data range: [{test_min:.6f}, {test_max:.6f}]")
        
        # Check if data is normalized
        train_mean = np.mean(X_train)
        train_std = np.std(X_train)
        
        print(f"📊 Training data mean: {train_mean:.6f}")
        print(f"📊 Training data std: {train_std:.6f}")
        
        # Check for normalization patterns
        if abs(train_mean) < 0.1 and abs(train_std - 1.0) < 0.1:
            print("✅ Data appears to be z-score normalized")
        elif train_min >= 0 and train_max <= 1:
            print("✅ Data appears to be min-max normalized")
        else:
            self.warnings.append("⚠️  Data doesn't appear to be normalized")
        
        # Check for constant features
        constant_features = np.sum(np.std(X_train, axis=0) < 1e-8)
        if constant_features > 0:
            self.warnings.append(f"⚠️  {constant_features} features appear to be constant")
        else:
            print("✅ No constant features detected")
    
    def validate_sample_integrity(self):
        """Check sample-level data integrity"""
        print(f"\n🔬 SAMPLE INTEGRITY VALIDATION")
        print("-" * 30)
        
        X_train = self.data['X_train']
        X_test = self.data['X_test']
        
        # Check for identical samples (potential data leakage)
        print("🔍 Checking for duplicate samples...")
        
        # Sample a subset for duplicate checking (full check would be too slow)
        sample_size = min(1000, X_train.shape[0])
        train_sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
        test_sample_indices = np.random.choice(X_test.shape[0], min(sample_size, X_test.shape[0]), replace=False)
        
        train_sample = X_train[train_sample_indices]
        test_sample = X_test[test_sample_indices]
        
        # Check for duplicates within training set
        unique_train = np.unique(train_sample, axis=0)
        train_duplicates = sample_size - len(unique_train)
        
        if train_duplicates > 0:
            self.warnings.append(f"⚠️  Found {train_duplicates} duplicate samples in training set sample")
        else:
            print("✅ No duplicates found in training set sample")
        
        # Check for train-test leakage (simplified check)
        print("🔍 Checking for potential train-test leakage...")
        
        # Use a simplified check on statistics rather than exact matches
        train_means = np.mean(train_sample, axis=1)
        test_means = np.mean(test_sample, axis=1)
        
        # Check if any test samples have very similar means to training samples
        potential_leaks = 0
        for test_mean in test_means[:100]:  # Check first 100 test samples
            min_diff = np.min(np.abs(train_means - test_mean))
            if min_diff < 1e-10:  # Very small difference suggests potential duplicate
                potential_leaks += 1
        
        if potential_leaks > 0:
            self.warnings.append(f"⚠️  {potential_leaks} test samples have very similar statistics to training samples")
        else:
            print("✅ No obvious train-test leakage detected")
    
    def generate_summary_plots(self, save_plots=True):
        """Generate summary visualizations"""
        print(f"\n📊 GENERATING SUMMARY PLOTS")
        print("-" * 30)
        
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Validation Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Label distribution
        ax1 = axes[0, 0]
        labels = ['ECG', 'EEG']
        train_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
        test_counts = [np.sum(self.data['y_test'] == 0), np.sum(self.data['y_test'] == 1)]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax1.bar(x - width/2, train_counts, width, label='Training', alpha=0.8)
        ax1.bar(x + width/2, test_counts, width, label='Test', alpha=0.8)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Sample Count')
        ax1.set_title('Class Distribution')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature statistics distribution
        ax2 = axes[0, 1]
        feature_means = np.mean(X_train, axis=0)
        ax2.hist(feature_means, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Feature Mean')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Feature Means')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sample statistics
        ax3 = axes[1, 0]
        sample_means = np.mean(X_train, axis=1)
        ecg_means = sample_means[y_train == 0]
        eeg_means = sample_means[y_train == 1]
        
        ax3.hist(ecg_means, bins=50, alpha=0.7, label='ECG', density=True)
        ax3.hist(eeg_means, bins=50, alpha=0.7, label='EEG', density=True)
        ax3.set_xlabel('Sample Mean')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Sample Means by Class')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature variance
        ax4 = axes[1, 1]
        feature_vars = np.var(X_train, axis=0)
        ax4.hist(feature_vars, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Feature Variance')
        ax4.set_ylabel('Count')
        ax4.set_title('Distribution of Feature Variances')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = 'dataset_validation_plots.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to: {plot_path}")
        
        plt.show()
    
    def generate_final_report(self):
        """Generate final validation report"""
        print(f"\n📋 FINAL VALIDATION REPORT")
        print("=" * 60)
        
        # Summary statistics
        print(f"📊 Dataset Summary:")
        print(f"   • Total samples: {self.stats['total_samples']:,}")
        print(f"   • Training samples: {self.stats['train_samples']:,}")
        print(f"   • Test samples: {self.stats['test_samples']:,}")
        print(f"   • Features: {self.stats['features']:,}")
        print(f"   • File size: {os.path.getsize(self.dataset_path) / 1024**2:.1f} MB")
        
        # Issues found
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES FOUND ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   {issue}")
            print(f"\n🛑 RECOMMENDATION: Fix these issues before training!")
            return False
        else:
            print(f"\n✅ NO CRITICAL ISSUES FOUND")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   {warning}")
            print(f"\n💡 These warnings should be reviewed but don't prevent training")
        
        # Final verdict
        print(f"\n🎯 FINAL VERDICT:")
        if not self.issues:
            if not self.warnings:
                print("   🟢 EXCELLENT: Dataset is ready for training!")
            elif len(self.warnings) <= 2:
                print("   🟡 GOOD: Dataset is ready with minor warnings")
            else:
                print("   🟠 CAUTION: Dataset has several warnings - review recommended")
            print(f"\n🚀 You can proceed with training!")
            return True
        else:
            print("   🔴 CRITICAL ISSUES: Dataset needs fixes before training")
            return False
    
    def run_full_validation(self, generate_plots=True):
        """Run complete dataset validation"""
        
        # Load dataset
        if not self.load_dataset():
            return False
        
        # Run all validations
        self.validate_structure()
        self.validate_shapes()
        self.validate_labels()
        self.validate_data_quality()
        self.validate_sample_integrity()
        
        # Generate plots
        if generate_plots:
            self.generate_summary_plots()
        
        # Final report
        return self.generate_final_report()

def main():
    """Main validation function"""
    
    # Check if dataset exists
    dataset_path = 'data/preprocessed_dataset.npz'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print(f"💡 Make sure you've run the data preparation first:")
        print(f"   python train.py --reload")
        return
    
    # Run validation
    validator = DatasetValidator(dataset_path)
    
    try:
        is_valid = validator.run_full_validation(generate_plots=True)
        
        if is_valid:
            print(f"\n🎉 VALIDATION PASSED!")
            print(f"Ready to start training with:")
            print(f"   python train.py all --learning-curve --quick-lc")
        else:
            print(f"\n🛑 VALIDATION FAILED!")
            print(f"Please fix the issues above before training.")
            
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()