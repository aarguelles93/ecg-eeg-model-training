# enhanced_sanity_check.py
"""
Enhanced sanity check for ECG vs EEG classification models
Reuses existing infrastructure and adds comprehensive debugging capabilities
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_curve, roc_curve, auc, log_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Reuse existing infrastructure
from data_loader import prepare_dataset, normalize_datasets, debug_data_after_normalization
from training_utils import MemoryManager, monitor_memory
from config import CONFIG

class ModelSanityChecker:
    """
    Comprehensive model evaluation and debugging tool
    """
    
    def __init__(self, output_dir="sanity_check_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Track evaluation results for comparison
        self.evaluation_results = {}
        self.data_diagnostics = {}
        
        print(f"🔍 Sanity Checker initialized - Results will be saved to: {output_dir}")
    
    def diagnose_data_preprocessing(self, X_train, X_test, y_train, y_test, metadata):
        """
        Comprehensive data preprocessing diagnostics to identify potential issues
        """
        print("\n🔬 COMPREHENSIVE DATA DIAGNOSTICS")
        print("=" * 60)
        
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'train_labels': np.bincount(y_train).tolist(),
            'test_labels': np.bincount(y_test).tolist(),
        }
        
        # Statistical analysis
        print("📊 Statistical Analysis:")
        for name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
            print(f"\n   {name} Dataset:")
            print(f"      Shape: {X.shape}")
            print(f"      Labels: {np.bincount(y)}")
            print(f"      Data range: [{X.min():.6f}, {X.max():.6f}]")
            print(f"      Mean: {X.mean():.6f}, Std: {X.std():.6f}")
            
            # Check for potential issues
            nan_count = np.sum(np.isnan(X))
            inf_count = np.sum(np.isinf(X))
            zero_count = np.sum(X == 0)
            extreme_count = np.sum(np.abs(X) > 10)
            
            print(f"      NaN values: {nan_count}")
            print(f"      Infinite values: {inf_count}")
            print(f"      Zero values: {zero_count} ({zero_count/X.size*100:.2f}%)")
            print(f"      Extreme values (>10): {extreme_count} ({extreme_count/X.size*100:.4f}%)")
            
            # Store diagnostics
            diagnostics[f'{name.lower()}_stats'] = {
                'mean': float(X.mean()),
                'std': float(X.std()),
                'min': float(X.min()),
                'max': float(X.max()),
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
                'zero_count': int(zero_count),
                'extreme_count': int(extreme_count)
            }
        
        # Feature distribution comparison
        print("\n📈 Feature Distribution Analysis:")
        self._analyze_feature_distributions(X_train, X_test)
        
        # Check train/test consistency
        print("\n🔄 Train/Test Consistency Check:")
        train_mean, test_mean = X_train.mean(), X_test.mean()
        train_std, test_std = X_train.std(), X_test.std()
        
        mean_diff = abs(train_mean - test_mean)
        std_diff = abs(train_std - test_std)
        
        print(f"   Mean difference: {mean_diff:.6f}")
        print(f"   Std difference: {std_diff:.6f}")
        
        if mean_diff > 0.1:
            print("   ⚠️  WARNING: Large mean difference between train/test!")
        if std_diff > 0.1:
            print("   ⚠️  WARNING: Large std difference between train/test!")
        
        diagnostics['consistency'] = {
            'mean_diff': float(mean_diff),
            'std_diff': float(std_diff),
            'mean_diff_warning': mean_diff > 0.1,
            'std_diff_warning': std_diff > 0.1
        }
        
        # EEG structure validation
        print("\n🧠 EEG Structure Validation:")
        expected_features = metadata.get('eeg_channels', 32) * metadata.get('eeg_timepoints', 188)
        actual_features = X_train.shape[1]
        
        print(f"   Expected features: {expected_features}")
        print(f"   Actual features: {actual_features}")
        print(f"   Structure valid: {metadata.get('structure_valid', False)}")
        
        if actual_features != expected_features:
            print("   ⚠️  WARNING: Feature count doesn't match expected EEG structure!")
        
        diagnostics['eeg_structure'] = {
            'expected_features': int(expected_features),
            'actual_features': int(actual_features),
            'structure_valid': bool(metadata.get('structure_valid', False))
        }
        
        self.data_diagnostics = diagnostics
        
        # Save diagnostics with JSON serialization fix
        diagnostics_path = os.path.join(self.output_dir, 'data_diagnostics.json')
        import json
        
        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert diagnostics to JSON-serializable format
        json_safe_diagnostics = convert_numpy_types(diagnostics)
        
        with open(diagnostics_path, 'w') as f:
            json.dump(json_safe_diagnostics, f, indent=2)
        
        print(f"\n💾 Data diagnostics saved to: {diagnostics_path}")
        
        return diagnostics
    
    def _analyze_feature_distributions(self, X_train, X_test, n_features_to_plot=10):
        """Analyze and plot feature distributions"""
        
        # Select random features to analyze
        n_features = min(n_features_to_plot, X_train.shape[1])
        feature_indices = np.random.choice(X_train.shape[1], n_features, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, feat_idx in enumerate(feature_indices):
            ax = axes[i]
            
            # Plot histograms
            ax.hist(X_train[:, feat_idx], bins=50, alpha=0.7, label='Train', density=True)
            ax.hist(X_test[:, feat_idx], bins=50, alpha=0.7, label='Test', density=True)
            ax.set_title(f'Feature {feat_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Feature distribution plot saved")
    
    def evaluate_model_comprehensive(self, model_path, X_test, y_test, model_name, 
                                   model_type='keras', input_preprocessing=None):
        """
        Comprehensive model evaluation with debugging information
        FIXED: Proper data reshaping based on EEG structure
        """
        print(f"\n🔍 COMPREHENSIVE EVALUATION: {model_name.upper()}")
        print("=" * 60)
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        # Memory monitoring
        monitor_memory(f"before {model_name} evaluation")
        
        try:
            # Load model
            if model_type == 'keras':
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
                print(f"📊 Model loaded successfully")
                print(f"   Input shape: {model.input_shape}")
                print(f"   Output shape: {model.output_shape}")
                print(f"   Parameters: {model.count_params():,}")
                
                # CRITICAL FIX: Proper data reshaping based on model input shape
                expected_shape = model.input_shape[1:]  # Remove batch dimension
                
                if len(expected_shape) == 2:  # (timepoints, channels) - for CNN/LSTM models
                    expected_timepoints, expected_channels = expected_shape
                    print(f"   Expected input structure: {expected_timepoints} timepoints × {expected_channels} channels")
                    
                    # Reshape from (batch, 6016) to (batch, timepoints, channels)
                    if X_test.shape[1] == expected_timepoints * expected_channels:
                        X_test_processed = X_test.reshape(-1, expected_timepoints, expected_channels)
                        print(f"   Reshaped to EEG structure: {X_test.shape} → {X_test_processed.shape}")
                    else:
                        print(f"   ⚠️  Feature count mismatch: expected {expected_timepoints * expected_channels}, got {X_test.shape[1]}")
                        print(f"   Using fallback reshaping...")
                        X_test_processed = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                        
                elif len(expected_shape) == 1:  # (features,) - for MLP models
                    expected_features = expected_shape[0]
                    print(f"   Expected input structure: {expected_features} features (MLP)")
                    
                    # Flatten for MLP
                    X_test_processed = X_test.reshape(X_test.shape[0], -1)
                    print(f"   Flattened for MLP: {X_test.shape} → {X_test_processed.shape}")
                    
                else:
                    print(f"   ⚠️  Unexpected input shape: {expected_shape}")
                    X_test_processed = X_test.copy()
                
            else:  # sklearn
                model = joblib.load(model_path)
                print(f"📊 SVM model loaded successfully")
                # Always flatten for sklearn models
                X_test_processed = X_test.reshape(X_test.shape[0], -1)
                print(f"   Flattened for SVM: {X_test.shape} → {X_test_processed.shape}")
            
            # Apply additional preprocessing if specified (legacy support)
            if input_preprocessing and model_type != 'keras':  # Only for non-keras models
                X_test_original = X_test_processed.copy()
                X_test_processed = input_preprocessing(X_test_processed)
                print(f"   Additional preprocessing applied: {X_test_original.shape} → {X_test_processed.shape}")
            
            # Input validation
            print(f"\n🔍 Input Validation:")
            print(f"   Test data shape: {X_test_processed.shape}")
            print(f"   Test data range: [{X_test_processed.min():.6f}, {X_test_processed.max():.6f}]")
            print(f"   Test data stats: mean={X_test_processed.mean():.6f}, std={X_test_processed.std():.6f}")
            
            # Check for problematic values
            nan_count = np.sum(np.isnan(X_test_processed))
            inf_count = np.sum(np.isinf(X_test_processed))
            if nan_count > 0 or inf_count > 0:
                print(f"   ⚠️  WARNING: Found {nan_count} NaN and {inf_count} infinite values!")
            
            # Validate shapes match what model expects
            if model_type == 'keras':
                if X_test_processed.shape[1:] != model.input_shape[1:]:
                    print(f"   ❌ SHAPE MISMATCH!")
                    print(f"      Model expects: {model.input_shape}")
                    print(f"      Data provides: {X_test_processed.shape}")
                    return None
                else:
                    print(f"   ✅ Shape validation passed!")
            
            # Generate predictions
            print(f"\n🎯 Generating Predictions:")
            if model_type == 'keras':
                # Use smaller batch size to avoid memory issues
                batch_size = min(32, len(X_test_processed))
                y_probs = model.predict(X_test_processed, verbose=0, batch_size=batch_size)
                
                # Handle different output shapes
                if y_probs.ndim > 1 and y_probs.shape[1] == 1:
                    y_probs = y_probs.flatten()
                elif y_probs.ndim > 1 and y_probs.shape[1] > 1:
                    # Multi-class: take probability of positive class
                    y_probs = y_probs[:, 1]
                
                y_pred = (y_probs > 0.5).astype(int)
                
                print(f"   Probability range: [{y_probs.min():.4f}, {y_probs.max():.4f}]")
                print(f"   Probability stats: mean={y_probs.mean():.4f}, std={y_probs.std():.4f}")
                
            else:  # SVM
                y_pred = model.predict(X_test_processed)
                try:
                    y_probs = model.predict_proba(X_test_processed)[:, 1]
                except:
                    y_probs = model.decision_function(X_test_processed)
                    # Normalize decision function to [0, 1] range
                    y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n📊 Performance Metrics:")
            print(f"   Accuracy: {accuracy:.4f}")
            
            # Detailed classification report
            print(f"\n📋 Classification Report:")
            report = classification_report(y_test, y_pred, target_names=["ECG", "EEG"], output_dict=True)
            print(classification_report(y_test, y_pred, target_names=["ECG", "EEG"]))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n📉 Confusion Matrix:")
            print(cm)
            
            # Probability calibration analysis
            if model_type == 'keras' or hasattr(model, 'predict_proba'):
                calibration_results = self._analyze_calibration(y_test, y_probs, model_name)
            else:
                calibration_results = None
            
            # Prediction analysis
            self._analyze_predictions(y_test, y_pred, y_probs, model_name)
            
            # Store results
            evaluation_result = {
                'model_name': model_name,
                'model_type': model_type,
                'accuracy': float(accuracy),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'input_shape': X_test_processed.shape,
                'prediction_stats': {
                    'y_pred_mean': float(y_pred.mean()),
                    'y_probs_mean': float(y_probs.mean()),
                    'y_probs_std': float(y_probs.std()),
                    'y_probs_range': [float(y_probs.min()), float(y_probs.max())]
                }
            }
            
            if calibration_results:
                evaluation_result['calibration'] = calibration_results
            
            self.evaluation_results[model_name] = evaluation_result
            
            print(f"✅ {model_name} evaluation completed successfully")
            
            # Memory cleanup
            del model
            if model_type == 'keras':
                import tensorflow as tf
                tf.keras.backend.clear_session()
            
            monitor_memory(f"after {model_name} evaluation")
            
            return evaluation_result
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_calibration(self, y_true, y_probs, model_name):
        """Analyze model calibration and create reliability diagram"""
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_probs, n_bins=10, strategy='uniform'
        )
        
        # Calculate calibration metrics
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Create reliability diagram
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2, label=f'{model_name}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Reliability Diagram - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Histogram of predicted probabilities
        plt.subplot(2, 2, 2)
        plt.hist(y_probs, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Probabilities')
        plt.grid(True, alpha=0.3)
        
        # ROC Curve
        plt.subplot(2, 2, 3)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        plt.subplot(2, 2, 4)
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'calibration_analysis_{model_name.lower().replace(" ", "_")}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Calibration analysis: Error = {calibration_error:.4f}, ROC-AUC = {roc_auc:.4f}")
        
        return {
            'calibration_error': float(calibration_error),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'mean_predicted_values': mean_predicted_value.tolist(),
            'fraction_of_positives': fraction_of_positives.tolist()
        }
    
    def _analyze_predictions(self, y_true, y_pred, y_probs, model_name):
        """Analyze prediction patterns for debugging"""
        
        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        n_misclassified = np.sum(misclassified_mask)
        
        print(f"\n🔍 Prediction Analysis:")
        print(f"   Total samples: {len(y_true)}")
        print(f"   Misclassified: {n_misclassified} ({n_misclassified/len(y_true)*100:.2f}%)")
        
        if n_misclassified > 0:
            # Analyze confidence of misclassified predictions
            misclassified_probs = y_probs[misclassified_mask]
            correct_probs = y_probs[~misclassified_mask]
            
            print(f"   Confidence in misclassified: {misclassified_probs.mean():.4f} ± {misclassified_probs.std():.4f}")
            print(f"   Confidence in correct: {correct_probs.mean():.4f} ± {correct_probs.std():.4f}")
            
            # Find high-confidence wrong predictions (potential issues)
            high_conf_wrong = np.sum((misclassified_mask) & ((y_probs > 0.8) | (y_probs < 0.2)))
            print(f"   High-confidence errors: {high_conf_wrong} ({high_conf_wrong/len(y_true)*100:.2f}%)")
            
            if high_conf_wrong > 0:
                print("   ⚠️  WARNING: Model is very confident about some wrong predictions!")
    
    def compare_models(self):
        """Generate comprehensive model comparison"""
        
        if not self.evaluation_results:
            print("❌ No evaluation results to compare!")
            return
        
        print(f"\n📊 MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        # Create comparison table
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            row = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision (ECG)': results['classification_report']['ECG']['precision'],
                'Recall (ECG)': results['classification_report']['ECG']['recall'],
                'F1 (ECG)': results['classification_report']['ECG']['f1-score'],
                'Precision (EEG)': results['classification_report']['EEG']['precision'],
                'Recall (EEG)': results['classification_report']['EEG']['recall'],
                'F1 (EEG)': results['classification_report']['EEG']['f1-score'],
            }
            
            if 'calibration' in results:
                row['Calibration Error'] = results['calibration']['calibration_error']
                row['ROC-AUC'] = results['calibration']['roc_auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        # Save comparison
        comparison_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
        
        # Find best models
        best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        print(f"\n🏆 Best Overall Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
        
        if 'ROC-AUC' in comparison_df.columns:
            best_roc = comparison_df.loc[comparison_df['ROC-AUC'].idxmax()]
            print(f"🏆 Best ROC-AUC: {best_roc['Model']} ({best_roc['ROC-AUC']:.4f})")
        
        # Create comparison visualization
        self._create_comparison_plots(comparison_df)
        
        return comparison_df
    
    def _create_comparison_plots(self, comparison_df):
        """Create comprehensive comparison visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score comparison for both classes
        x = np.arange(len(comparison_df))
        width = 0.35
        axes[0, 1].bar(x - width/2, comparison_df['F1 (ECG)'], width, label='ECG F1', alpha=0.8)
        axes[0, 1].bar(x + width/2, comparison_df['F1 (EEG)'], width, label='EEG F1', alpha=0.8)
        axes[0, 1].set_title('F1 Score Comparison by Class')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(comparison_df['Model'], rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # ROC-AUC comparison (if available)
        if 'ROC-AUC' in comparison_df.columns:
            axes[1, 0].bar(comparison_df['Model'], comparison_df['ROC-AUC'])
            axes[1, 0].set_title('ROC-AUC Comparison')
            axes[1, 0].set_ylabel('ROC-AUC')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Calibration error comparison (if available)
        if 'Calibration Error' in comparison_df.columns:
            axes[1, 1].bar(comparison_df['Model'], comparison_df['Calibration Error'])
            axes[1, 1].set_title('Calibration Error Comparison (Lower is Better)')
            axes[1, 1].set_ylabel('Calibration Error')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plots saved")
    
    def export_debugging_report(self):
        """Export comprehensive debugging report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f'debugging_report_{timestamp}.json')
        
        report = {
            'timestamp': timestamp,
            'data_diagnostics': self.data_diagnostics,
            'evaluation_results': self.evaluation_results,
            'summary': {
                'total_models_evaluated': len(self.evaluation_results),
                'data_issues_detected': self._detect_data_issues(),
                'model_issues_detected': self._detect_model_issues()
            }
        }
        
        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to native Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Convert report to JSON-serializable format
        json_safe_report = convert_numpy_types(report)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(json_safe_report, f, indent=2)
        
        print(f"\n📋 Comprehensive debugging report exported to: {report_path}")
        
        return report
    
    def _detect_data_issues(self):
        """Detect potential data issues"""
        issues = []
        
        if self.data_diagnostics:
            # Check consistency issues
            if self.data_diagnostics.get('consistency', {}).get('mean_diff_warning', False):
                issues.append("Large mean difference between train/test data")
            
            if self.data_diagnostics.get('consistency', {}).get('std_diff_warning', False):
                issues.append("Large std difference between train/test data")
            
            # Check for problematic values
            for dataset in ['train_stats', 'test_stats']:
                if dataset in self.data_diagnostics:
                    stats = self.data_diagnostics[dataset]
                    if stats.get('nan_count', 0) > 0:
                        issues.append(f"NaN values detected in {dataset}")
                    if stats.get('inf_count', 0) > 0:
                        issues.append(f"Infinite values detected in {dataset}")
                    if stats.get('extreme_count', 0) > 0:
                        issues.append(f"Extreme values detected in {dataset}")
        
        return issues
    
    def _detect_model_issues(self):
        """Detect potential model issues"""
        issues = []
        
        for model_name, results in self.evaluation_results.items():
            # Check for poor performance
            if results['accuracy'] < 0.7:
                issues.append(f"{model_name}: Low accuracy ({results['accuracy']:.3f})")
            
            # Check for poor calibration
            if 'calibration' in results:
                if results['calibration']['calibration_error'] > 0.1:
                    issues.append(f"{model_name}: Poor calibration (error: {results['calibration']['calibration_error']:.3f})")
        
        return issues


def main():
    """Enhanced main function using existing infrastructure with argument parsing"""
    import argparse
    
    # Parse command line arguments to match train.py parameters
    parser = argparse.ArgumentParser(description='Enhanced sanity check for ECG vs EEG classification models')
    
    # Dataset file arguments (NEW: configurable file paths)
    parser.add_argument('--ecg-csv', default=None,
                       help='Path to ECG CSV file (auto-detected from cache if available)')
    parser.add_argument('--eeg-csv', default=None,
                       help='Path to EEG CSV file (auto-detected from cache if available)')
    
    # Dataset preparation arguments (matching train.py)
    parser.add_argument('--normalization', choices=['smart', 'zscore', 'minmax', 'per_sample'], 
                       default='smart', help='Normalization method (default: smart)')
    parser.add_argument('--norm-strategy', choices=['combined', 'separate'], 
                       default='separate', help='Normalization strategy (default: separate)')
    parser.add_argument('--dataset-fraction', type=float, default=1.0,
                       help='Fraction of dataset to use (default: 1.0)')
    parser.add_argument('--memory-limit', type=float, default=3.5,
                       help='Memory limit in GB for dataset processing (default: 3.5GB)')
    parser.add_argument('--reload', action='store_true',
                       help='Force reload dataset (ignore cache)')
    
    # Sanity check specific arguments
    parser.add_argument('--models', nargs='+', 
                       default=['svm', 'simple_cnn', 'cnn_lstm', 'mlp', 'tcn', 'dual_branch'],
                       help='Models to evaluate (default: all models)')
    parser.add_argument('--output-dir', default='sanity_check_results',
                       help='Output directory for results (default: sanity_check_results)')
    parser.add_argument('--debug-misclassifications', action='store_true',
                       help='Enable detailed misclassification debugging')
    parser.add_argument('--n-debug-examples', type=int, default=10,
                       help='Number of misclassification examples to analyze (default: 10)')
    parser.add_argument('--auto-detect-params', action='store_true', default=True,
                       help='Auto-detect parameters from cached dataset (default: True)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # AUTO-DETECT PARAMETERS FROM CACHE (if requested and available)
    detected_params = None
    if args.auto_detect_params:
        # Check if the verbose parameter is supported
        import inspect
        sig = inspect.signature(quick_parameter_check)
        
        if 'verbose' in sig.parameters:
            detected_params = quick_parameter_check(verbose=False)
        else:
            # Fallback for original function signature
            print("🔍 Auto-detecting parameters from cached dataset...")
            detected_params = quick_parameter_check()
        
        if detected_params:
            print("🔍 Auto-detected parameters from cached dataset:")
            
            # Override arguments with detected parameters if not explicitly set
            if not any(arg.startswith('--normalization') for arg in sys.argv[1:]):
                args.normalization = detected_params.get('normalization', args.normalization)
                print(f"   🔧 Using detected normalization: {args.normalization}")
            
            if not any(arg.startswith('--norm-strategy') for arg in sys.argv[1:]):
                args.norm_strategy = detected_params.get('normalization_strategy', args.norm_strategy)
                print(f"   🔧 Using detected norm-strategy: {args.norm_strategy}")
            
            # Auto-detect file paths if not provided
            if args.ecg_csv is None:
                args.ecg_csv = detected_params.get('ecg_path', 'data/mitbih_train.csv')
                print(f"   📄 Using detected ECG file: {args.ecg_csv}")
            
            if args.eeg_csv is None:
                args.eeg_csv = detected_params.get('eeg_path', 'data/eeg_dataset_32.csv')
                print(f"   📄 Using detected EEG file: {args.eeg_csv}")
        else:
            print("⚠️  No cached parameters found - using defaults")
    
    # Set default file paths if still not set
    if args.ecg_csv is None:
        args.ecg_csv = 'data/mitbih_train.csv'
    if args.eeg_csv is None:
        args.eeg_csv = 'data/eeg_dataset_32.csv'
    
    # Validate file existence
    if not os.path.exists(args.ecg_csv):
        print(f"❌ ECG file not found: {args.ecg_csv}")
        print("💡 Available files in data/ directory:")
        if os.path.exists('data'):
            for f in os.listdir('data'):
                if 'mitbih' in f.lower() or 'ecg' in f.lower():
                    print(f"   📄 {os.path.join('data', f)}")
        return
    
    if not os.path.exists(args.eeg_csv):
        print(f"❌ EEG file not found: {args.eeg_csv}")
        print("💡 Available files in data/ directory:")
        if os.path.exists('data'):
            for f in os.listdir('data'):
                if 'eeg' in f.lower():
                    print(f"   📄 {os.path.join('data', f)}")
        return
    
    print("🚀 ENHANCED MODEL SANITY CHECK")
    print("=" * 70)
    print(f"🔧 Using parameters:")
    print(f"   ECG file: {args.ecg_csv}")
    print(f"   EEG file: {args.eeg_csv}")
    print(f"   Normalization: {args.normalization} (strategy: {args.norm_strategy})")
    print(f"   Dataset fraction: {args.dataset_fraction*100:.1f}%")
    print(f"   Memory limit: {args.memory_limit}GB")
    print(f"   Models to evaluate: {', '.join(args.models)}")
    print(f"   Force reload: {args.reload}")
    if detected_params:
        print(f"   📋 Parameters auto-detected from cache")
    
    # Initialize checker with custom output directory
    checker = ModelSanityChecker(output_dir=args.output_dir)
    
    # Prepare dataset using SAME parameters as potentially used in training
    print("📦 Preparing dataset with MATCHING parameters...")
    
    # Use the EXACT same parameters that might have been used in training
    X_train, X_test, y_train, y_test, metadata = prepare_dataset(
        args.ecg_csv, args.eeg_csv,  # Use detected/specified file paths
        normalization=args.normalization,
        normalization_strategy=args.norm_strategy,
        validate_alignment=True,
        force_reload=args.reload,
        memory_limit_gb=args.memory_limit,
        dataset_fraction=args.dataset_fraction
    )
    
    print("✅ Dataset prepared successfully")
    print(f"   Parameters used: norm={args.normalization}, strategy={args.norm_strategy}, fraction={args.dataset_fraction}")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Features: {X_train.shape[1]:,}")
    
    # Comprehensive data diagnostics
    checker.diagnose_data_preprocessing(X_train, X_test, y_train, y_test, metadata)
    
    # Model evaluation - only evaluate requested models
    model_dir = 'models'
    
    # Define all possible models with their configurations (SIMPLIFIED - let the function handle reshaping)
    all_model_configs = {
        'svm': {
            'name': 'SVM',
            'path': os.path.join(model_dir, 'svm_model.joblib'),
            'type': 'sklearn'
        },
        'simple_cnn': {
            'name': 'Simple CNN',
            'path': os.path.join(model_dir, 'simple_cnn_final.keras'),
            'type': 'keras'
        },
        'cnn_lstm': {
            'name': 'CNN-LSTM',
            'path': os.path.join(model_dir, 'cnn_lstm_final.keras'),
            'type': 'keras'
        },
        'mlp': {
            'name': 'MLP',
            'path': os.path.join(model_dir, 'mlp_final.keras'),
            'type': 'keras'
        },
        'tcn': {
            'name': 'TCN',
            'path': os.path.join(model_dir, 'tcn_final.keras'),
            'type': 'keras'
        },
        'dual_branch': {
            'name': 'Dual-Branch',
            'path': os.path.join(model_dir, 'dual_branch_final.keras'),
            'type': 'keras'
        }
    }
    
    # Filter to only requested models
    model_configs = []
    for model_name in args.models:
        if model_name in all_model_configs:
            model_configs.append(all_model_configs[model_name])
        else:
            print(f"⚠️  Warning: Unknown model '{model_name}' - skipping")
    
    print(f"\n📊 Evaluating {len(model_configs)} models: {', '.join([cfg['name'] for cfg in model_configs])}")
    
    # Evaluate each model
    successful_evaluations = 0
    for model_config in model_configs:
        result = checker.evaluate_model_comprehensive(
            model_path=model_config['path'],
            X_test=X_test,
            y_test=y_test,
            model_name=model_config['name'],
            model_type=model_config['type']
        )
        
        if result is not None:
            successful_evaluations += 1
            
            # Optional: Debug misclassifications for this model
            if args.debug_misclassifications:
                debug_model_misclassifications(
                    model_path=model_config['path'],
                    X_test=X_test,
                    y_test=y_test,
                    model_name=model_config['name'],
                    output_dir=args.output_dir,
                    n_examples=args.n_debug_examples
                )
    
    print(f"\n📊 Evaluation Summary:")
    print(f"   Total models requested: {len(model_configs)}")
    print(f"   Successfully evaluated: {successful_evaluations}")
    print(f"   Failed evaluations: {len(model_configs) - successful_evaluations}")
    
    # Generate comprehensive comparison
    if checker.evaluation_results:
        comparison_df = checker.compare_models()
        
        # Export debugging report
        report = checker.export_debugging_report()
        
        # Print potential issues summary
        data_issues = report['summary']['data_issues_detected']
        model_issues = report['summary']['model_issues_detected']
        
        print(f"\n🔍 ISSUE DETECTION SUMMARY:")
        print(f"   Data issues: {len(data_issues)}")
        for issue in data_issues:
            print(f"     ⚠️  {issue}")
        
        print(f"   Model issues: {len(model_issues)}")
        for issue in model_issues:
            print(f"     ⚠️  {issue}")
        
        if not data_issues and not model_issues:
            print("   ✅ No significant issues detected!")
        
        # Enhanced recommendations for deployment based on detected parameters
        print(f"\n💡 DEPLOYMENT RECOMMENDATIONS:")
        print("=" * 50)
        print(f"📋 CRITICAL: Ensure deployment uses EXACT same parameters:")
        print(f"   • Normalization: {args.normalization}")
        print(f"   • Normalization strategy: {args.norm_strategy}")
        print(f"   • Dataset fraction: {args.dataset_fraction} (if subsampling was used)")
        print(f"   • Memory limit: {args.memory_limit}GB (affects chunking)")
        
        if data_issues:
            print("   🔧 Data Issues:")
            print("     • Ensure consistent preprocessing between training and deployment")
            print("     • Verify normalization parameters are saved and applied correctly")
            print("     • Check for data leakage or distribution shift")
        
        if model_issues:
            print("   🔧 Model Issues:")
            print("     • Consider model retraining or hyperparameter tuning")
            print("     • Implement probability calibration for better confidence estimates")
            print("     • Add uncertainty quantification for critical applications")
        
        print("   📋 General Recommendations:")
        print("     • Save and load exact preprocessing pipeline used during training")
        print("     • Implement input validation in deployment environment")
        print("     • Monitor model predictions for distribution drift")
        print("     • Use ensemble methods for improved robustness")
        print("     • Implement A/B testing for model deployment validation")
        
        # Create deployment checklist with parameter information
        create_deployment_checklist(args.output_dir, data_issues, model_issues, args)
        
    else:
        print("❌ No models were successfully evaluated!")
    
    print(f"\n🎉 Sanity check completed!")
    print(f"📁 All results saved to: {args.output_dir}")
    print(f"\n💡 To run with same parameters as training, use:")
    print(f"   python enhanced_sanity_check.py --normalization {args.normalization} --norm-strategy {args.norm_strategy} --dataset-fraction {args.dataset_fraction}")


def create_deployment_checklist(output_dir, data_issues, model_issues, args):
    """Create a deployment checklist based on detected issues and used parameters"""
    
    checklist_content = f"""# Model Deployment Checklist
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Training Parameters Used: normalization={args.normalization}, strategy={args.norm_strategy}, fraction={args.dataset_fraction}

## CRITICAL: Training Parameters Reproduction

### Exact Parameter Matching (MUST MATCH TRAINING)
- [ ] Normalization method: {args.normalization}
- [ ] Normalization strategy: {args.norm_strategy}
- [ ] Dataset fraction: {args.dataset_fraction} {"(subsampling was used!)" if args.dataset_fraction < 1.0 else "(full dataset)"}
- [ ] Memory limit: {args.memory_limit}GB (affects chunking behavior)
- [ ] Force reload setting: {args.reload}

### Critical Preprocessing Pipeline Validation
- [ ] Verify prepare_dataset() uses IDENTICAL parameters
- [ ] Confirm smart normalization auto-detection works identically
- [ ] Test feature compatibility check produces same result
- [ ] Validate EEG structure projection is identical
- [ ] Ensure class balancing is applied consistently

## Pre-Deployment Validation

### Data Pipeline Validation
- [ ] Verify input data format matches training data exactly
- [ ] Confirm normalization parameters are identical to training
- [ ] Test with edge cases (min/max values, missing data)
- [ ] Validate data preprocessing pipeline end-to-end
- [ ] Check for any data leakage or temporal inconsistencies

### Model Validation
- [ ] Load model successfully in target environment
- [ ] Verify model architecture matches training configuration
- [ ] Test model predictions on known validation samples
- [ ] Confirm output format and probability interpretation
- [ ] Validate model calibration if using probability thresholds

### Environment Validation
- [ ] Verify TensorFlow/scikit-learn versions match training environment
- [ ] Test on target hardware (CPU/GPU compatibility)
- [ ] Validate memory usage and performance requirements
- [ ] Test error handling and edge cases
- [ ] Implement proper logging and monitoring

## Detected Issues to Address

### Data Issues ({len(data_issues)} found):
"""
    
    for issue in data_issues:
        checklist_content += f"- [ ] Fix: {issue}\n"
    
    checklist_content += f"""
### Model Issues ({len(model_issues)} found):
"""
    
    for issue in model_issues:
        checklist_content += f"- [ ] Fix: {issue}\n"
    
    checklist_content += f"""
## Parameter-Specific Recommendations

### Normalization: {args.normalization}
"""
    
    if args.normalization == 'smart':
        checklist_content += """- [ ] Ensure deployment environment can auto-detect existing normalization
- [ ] Test that smart detection produces identical results in production
- [ ] Verify normalization state is preserved correctly
- [ ] Monitor for normalization detection failures
"""
    elif args.normalization == 'per_sample':
        checklist_content += """- [ ] Confirm per-sample normalization with 5σ clipping is applied
- [ ] Verify clip threshold matches training (5.0 or custom value)
- [ ] Test edge cases with extreme outliers
- [ ] Monitor for samples with zero variance
"""
    
    if args.dataset_fraction < 1.0:
        checklist_content += f"""
### Dataset Subsampling: {args.dataset_fraction}
- [ ] WARNING: Training used only {args.dataset_fraction*100:.1f}% of available data
- [ ] Verify this was intentional and not due to memory constraints
- [ ] Consider retraining on full dataset if resources allow
- [ ] Monitor for performance degradation due to limited training data
"""
    
    checklist_content += """
## Post-Deployment Monitoring

### Performance Monitoring
- [ ] Set up accuracy monitoring on production data
- [ ] Implement prediction confidence tracking
- [ ] Monitor for data distribution drift
- [ ] Track model latency and resource usage
- [ ] Set up alerting for performance degradation

### Data Quality Monitoring
- [ ] Monitor input data statistics
- [ ] Detect outliers and anomalous inputs
- [ ] Track feature importance changes
- [ ] Monitor for missing or corrupted data
- [ ] Implement data validation pipelines

### Model Health Monitoring
- [ ] Track prediction distribution changes
- [ ] Monitor for bias in predictions
- [ ] Implement A/B testing framework
- [ ] Set up model rollback procedures
- [ ] Plan for periodic model retraining

## Emergency Procedures

### Model Failure Response
- [ ] Document rollback procedures
- [ ] Implement fallback model or rule-based system
- [ ] Define escalation procedures
- [ ] Test disaster recovery scenarios
- [ ] Maintain model versioning and rollback capability

### Data Pipeline Failure Response
- [ ] Implement data validation checks
- [ ] Define data quality thresholds
- [ ] Set up data pipeline monitoring
- [ ] Document data recovery procedures
- [ ] Test with degraded data scenarios

## Sign-off

- [ ] Data Science Team Approval
- [ ] Engineering Team Approval
- [ ] QA Team Approval
- [ ] Product Team Approval
- [ ] Security Team Approval (if applicable)

---
Generated by Enhanced Sanity Check Tool
Command used: python enhanced_sanity_check.py --normalization {args.normalization} --norm-strategy {args.norm_strategy} --dataset-fraction {args.dataset_fraction}
"""
    
    checklist_path = os.path.join(output_dir, 'deployment_checklist.md')
    with open(checklist_path, 'w') as f:
        f.write(checklist_content)
    
    print(f"📋 Deployment checklist created: {checklist_path}")


def quick_parameter_check():
    """Quick utility to check what parameters were likely used in training"""
    
    print("🔍 TRAINING PARAMETER DETECTIVE")
    print("=" * 40)
    
    # Check for cached dataset
    cache_path = 'data/preprocessed_dataset.npz'
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            if 'cache_params' in cached:
                cache_params = cached['cache_params'].item()
                print("✅ Found cached dataset with parameters:")
                for key, value in cache_params.items():
                    print(f"   {key}: {value}")
                print(f"\n💡 Use these parameters for sanity check:")
                print(f"   python enhanced_sanity_check.py \\")
                print(f"     --normalization {cache_params.get('normalization', 'smart')} \\")
                print(f"     --norm-strategy {cache_params.get('normalization_strategy', 'separate')}")
                return cache_params
            else:
                print("⚠️  Found cached dataset but no parameter information")
        except Exception as e:
            print(f"⚠️  Could not read cached dataset: {e}")
    else:
        print("❌ No cached dataset found")
        print("💡 Run train.py first, or use default parameters")
    
    return None


def debug_model_misclassifications(model_path, X_test, y_test, model_name, 
                                 output_dir="sanity_check_results", n_examples=10):
    """
    Debug specific misclassifications to understand model behavior
    """
    print(f"\n🔍 DEBUGGING MISCLASSIFICATIONS: {model_name}")
    print("=" * 60)
    
    try:
        # Load model
        if model_path.endswith('.keras'):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            
            # Prepare input based on model type
            if 'mlp' in model_name.lower():
                X_test_input = X_test.reshape(X_test.shape[0], -1)
            else:
                X_test_input = X_test.reshape(X_test.shape[0], X_test.shape[1], 1) if len(X_test.shape) == 2 else X_test
            
            y_probs = model.predict(X_test_input, verbose=0)
            if y_probs.ndim > 1 and y_probs.shape[1] == 1:
                y_probs = y_probs.flatten()
            
        else:  # SVM
            model = joblib.load(model_path)
            X_test_input = X_test.reshape(X_test.shape[0], -1)
            y_probs = model.predict_proba(X_test_input)[:, 1]
        
        y_pred = (y_probs > 0.5).astype(int)
        
        # Find misclassified examples
        misclassified_mask = y_test != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        print(f"📊 Found {len(misclassified_indices)} misclassified samples")
        
        if len(misclassified_indices) == 0:
            print("✅ Perfect classification - no misclassifications to debug!")
            return
        
        # Analyze confidence distribution of errors
        error_confidences = y_probs[misclassified_mask]
        high_conf_errors = np.sum((error_confidences > 0.8) | (error_confidences < 0.2))
        
        print(f"📈 Error Analysis:")
        print(f"   High-confidence errors: {high_conf_errors}/{len(misclassified_indices)} ({high_conf_errors/len(misclassified_indices)*100:.1f}%)")
        print(f"   Mean error confidence: {error_confidences.mean():.3f} ± {error_confidences.std():.3f}")
        
        # Sample worst errors for detailed analysis
        n_examples = min(n_examples, len(misclassified_indices))
        
        # Sort by confidence (most confident wrong predictions first)
        error_confidence_scores = np.abs(y_probs[misclassified_mask] - 0.5)  # Distance from uncertain
        worst_error_indices = misclassified_indices[np.argsort(error_confidence_scores)[-n_examples:]]
        
        print(f"\n🔬 Analyzing {n_examples} worst misclassifications:")
        
        analysis_results = []
        for i, idx in enumerate(worst_error_indices):
            true_label = "EEG" if y_test[idx] == 1 else "ECG"
            pred_label = "EEG" if y_pred[idx] == 1 else "ECG"
            confidence = y_probs[idx]
            
            # Analyze signal characteristics
            signal = X_test[idx]
            signal_stats = {
                'mean': signal.mean(),
                'std': signal.std(),
                'min': signal.min(),
                'max': signal.max(),
                'range': signal.max() - signal.min(),
                'energy': np.sum(signal**2),
                'zero_crossings': np.sum(np.diff(np.sign(signal)) != 0)
            }
            
            print(f"\n   Example {i+1} (Sample #{idx}):")
            print(f"     True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.3f}")
            print(f"     Signal stats: mean={signal_stats['mean']:.3f}, std={signal_stats['std']:.3f}")
            print(f"     Range: [{signal_stats['min']:.3f}, {signal_stats['max']:.3f}]")
            print(f"     Energy: {signal_stats['energy']:.1f}, Zero crossings: {signal_stats['zero_crossings']}")
            
            analysis_results.append({
                'sample_idx': int(idx),
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': float(confidence),
                'signal_stats': signal_stats
            })
        
        # Save detailed analysis
        import json
        analysis_path = os.path.join(output_dir, f'misclassification_analysis_{model_name.lower().replace(" ", "_")}.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\n💾 Detailed analysis saved to: {analysis_path}")
        
        # Create visualization of misclassified samples
        create_misclassification_plots(X_test, y_test, y_pred, y_probs, worst_error_indices, 
                                     model_name, output_dir)
        
    except Exception as e:
        print(f"❌ Error debugging {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()


def create_misclassification_plots(X_test, y_test, y_pred, y_probs, error_indices, 
                                 model_name, output_dir):
    """Create visualization plots for misclassified samples"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(error_indices[:9]):  # Plot up to 9 examples
        ax = axes[i]
        
        signal = X_test[idx]
        true_label = "EEG" if y_test[idx] == 1 else "ECG"
        pred_label = "EEG" if y_pred[idx] == 1 else "ECG"
        confidence = y_probs[idx]
        
        # Plot signal
        ax.plot(signal, linewidth=0.8)
        ax.set_title(f'Sample {idx}\nTrue: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time/Feature Index')
        ax.set_ylabel('Amplitude')
        
        # Color-code by error type
        if true_label != pred_label:
            ax.patch.set_facecolor('red' if confidence > 0.7 or confidence < 0.3 else 'orange')
            ax.patch.set_alpha(0.1)
    
    # Hide unused subplots
    for i in range(len(error_indices), 9):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'misclassified_signals_{model_name.lower().replace(" ", "_")}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Misclassification plots saved")


if __name__ == "__main__":
    import sys
    
    # Special case: if called with --check-params, run parameter detective
    if '--check-params' in sys.argv:
        quick_parameter_check()
    else:
        main()
