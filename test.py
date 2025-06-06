"""
Modern test.py - Enhanced Model Evaluation for ECG vs EEG Classification
UPDATED: Compatible with new data pipeline, cached datasets, and memory management
"""

import os
import gc
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Import our modern utilities
from training_utils import cleanup_memory, monitor_memory, create_optimizer
from gpu_config import monitor_gpu_memory

# Sklearn metrics
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score
)

# TensorFlow imports (safe after training_utils)
from tensorflow.keras.models import load_model
import tensorflow as tf

# Data loading with modern pipeline
from data_loader import prepare_dataset

# Global configuration
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
DATA_DIR = 'data'

# Results subdirectories
RESULTS_STRUCTURE = {
    'confusion_matrices': 'confusion_matrices',
    'roc_curves': 'roc_curves', 
    'metrics': 'metrics',
    'comparisons': 'comparisons',
    'detailed_reports': 'detailed_reports'
}

class ModelEvaluator:
    """
    Modern model evaluation class with cached dataset support,
    memory management, and comprehensive metrics
    """
    
    def __init__(self, results_dir=RESULTS_DIR, models_dir=MODELS_DIR):
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        
        # Create organized results structure
        self.results_structure = {}
        for key, subdir in RESULTS_STRUCTURE.items():
            full_path = self.results_dir / subdir
            full_path.mkdir(parents=True, exist_ok=True)
            self.results_structure[key] = full_path
        
        # Track evaluation results
        self.all_results = {}
        self.evaluation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🎯 ModelEvaluator initialized")
        print(f"   📁 Results dir: {self.results_dir}")
        print(f"   📁 Models dir: {self.models_dir}")
        print(f"   📊 Results structure:")
        for key, path in self.results_structure.items():
            print(f"      {key}: {path.relative_to(self.results_dir)}/")
        
        # Also ensure main results directory exists
        self.results_dir.mkdir(exist_ok=True)
    
    def find_available_models(self):
        """
        Automatically discover available trained models
        Returns dict of {model_name: model_path}
        """
        model_patterns = {
            'svm': ['svm_model.joblib', 'svm_final.joblib'],
            'simple_cnn': ['simple_cnn_best.keras', 'simple_cnn_final.keras'],
            'cnn_lstm': ['cnn_lstm_best.keras', 'cnn_lstm_final.keras'], 
            'mlp': ['mlp_best.keras', 'mlp_final.keras'],
            'tcn': ['tcn_best.keras', 'tcn_final.keras'],
            'dual_branch': ['dual_branch_best.keras', 'dual_branch_final.keras']
        }
        
        available_models = {}
        
        print("🔍 Searching for trained models...")
        
        for model_name, patterns in model_patterns.items():
            for pattern in patterns:
                model_path = self.models_dir / pattern
                if model_path.exists():
                    available_models[model_name] = model_path
                    print(f"   ✅ Found {model_name}: {model_path}")
                    break
            else:
                print(f"   ❌ {model_name}: No trained model found")
        
        if not available_models:
            print("⚠️  No trained models found! Run training first.")
        
        return available_models
    
    def load_cached_dataset(self, force_reload=False, **dataset_kwargs):
        """
        Load dataset using the modern cached system from data_loader.py
        Uses same logic as training pipeline
        """
        print("📦 Loading test dataset with modern pipeline...")
        
        # Use same paths as training
        ecg_path = Path(DATA_DIR) / 'mitbih_train.csv'
        eeg_path = Path(DATA_DIR) / 'eeg_dataset_32.csv'  # Updated path
        
        # Check if paths exist
        if not ecg_path.exists():
            raise FileNotFoundError(f"ECG dataset not found: {ecg_path}")
        if not eeg_path.exists():
            raise FileNotFoundError(f"EEG dataset not found: {eeg_path}")
        
        print(f"   📁 ECG data: {ecg_path}")
        print(f"   📁 EEG data: {eeg_path}")
        
        # Load with same parameters as training (cached by default)
        dataset_params = {
            'normalization': 'smart',
            'normalization_strategy': 'separate', 
            'validate_alignment': True,
            'force_reload': force_reload,
            **dataset_kwargs
        }
        
        print(f"   🔧 Dataset parameters: {dataset_params}")
        
        # Load dataset
        start_time = time.time()
        X_train, X_test, y_train, y_test, metadata = prepare_dataset(
            str(ecg_path), str(eeg_path), **dataset_params
        )
        load_time = time.time() - start_time
        
        print(f"✅ Dataset loaded in {load_time:.1f}s")
        print(f"   📊 Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"   🏷️  Train labels: {np.bincount(y_train)}, Test labels: {np.bincount(y_test)}")
        print(f"   📈 Feature range: [{X_test.min():.3f}, {X_test.max():.3f}]")
        print(f"   🧠 EEG structure: {metadata.get('eeg_channels', 'unknown')}ch × {metadata.get('eeg_timepoints', 'unknown')}tp")
        
        return X_train, X_test, y_train, y_test, metadata
    
    def prepare_model_data(self, X_test, model_name, metadata):
        """
        Prepare test data with correct shape for different model types
        Uses the same logic as training pipeline
        """
        print(f"🔧 Preparing test data for {model_name}...")
        print(f"   Original shape: {X_test.shape}")
        
        if model_name == 'mlp':
            # MLP needs flattened input
            X_test_model = X_test.reshape(X_test.shape[0], -1)
            input_shape = (X_test_model.shape[1],)
            print(f"   MLP flattened shape: {X_test_model.shape}")
            
        else:
            # Use EEG structure for proper reshaping
            channels = metadata.get('eeg_channels', 32)
            timepoints = metadata.get('eeg_timepoints', 188)
            expected_features = channels * timepoints
            
            print(f"   EEG structure: {channels}ch × {timepoints}tp = {expected_features} features")
            
            if X_test.shape[1] != expected_features:
                print(f"   ⚠️  Shape mismatch! Expected {expected_features}, got {X_test.shape[1]}")
                # Fallback: treat as (samples, features, 1)
                X_test_model = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                input_shape = (X_test.shape[1], 1)
            else:
                # Proper EEG reshaping: (samples, timepoints, channels)
                X_test_model = X_test.reshape(-1, timepoints, channels)
                input_shape = (timepoints, channels)
                print(f"   Reshaped to EEG structure: {X_test_model.shape}")
        
        return X_test_model, input_shape
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=True):
        """Enhanced confusion matrix with additional statistics"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Main confusion matrix
        ax1 = plt.subplot(2, 2, (1, 2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=["ECG", "EEG"], yticklabels=["ECG", "EEG"], ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title(f'Confusion Matrix - {model_name}')
        
        # Normalized confusion matrix
        ax2 = plt.subplot(2, 2, 3)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', cbar=True,
                    xticklabels=["ECG", "EEG"], yticklabels=["ECG", "EEG"], ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('Normalized Confusion Matrix')
        
        # Performance summary
        ax3 = plt.subplot(2, 2, 4)
        ax3.axis('off')
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        summary_text = f"""
Model: {model_name}
───────────────────
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1-Score:  {f1:.4f}
MCC:       {mcc:.4f}
Cohen's κ: {kappa:.4f}

Test Samples: {len(y_true):,}
ECG Samples:  {np.sum(y_true == 0):,}
EEG Samples:  {np.sum(y_true == 1):,}
        """
        
        ax3.text(0.1, 0.5, summary_text, fontsize=11, fontfamily='monospace',
                 verticalalignment='center', transform=ax3.transAxes)
        
        plt.tight_layout()
        
        if save:
            save_path = self.results_structure['confusion_matrices'] / f'{model_name}_confusion_matrix_{self.evaluation_timestamp}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📊 Confusion matrix saved: {save_path.relative_to(self.results_dir)}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true, y_probs, model_name, save=True):
        """Enhanced ROC curve with precision-recall curve"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {model_name}')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        
        ax2.plot(recall, precision, color='red', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.3f})')
        ax2.axhline(y=np.mean(y_true), color='navy', linestyle='--', alpha=0.5,
                    label=f'Baseline ({np.mean(y_true):.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {model_name}')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.results_structure['roc_curves'] / f'{model_name}_curves_{self.evaluation_timestamp}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   📈 ROC/PR curves saved: {save_path.relative_to(self.results_dir)}")
        
        plt.close()
        
        return roc_auc, pr_auc
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_probs=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        }
        
        if y_probs is not None:
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            precision, recall, _ = precision_recall_curve(y_true, y_probs)
            metrics.update({
                'roc_auc': auc(fpr, tpr),
                'pr_auc': auc(recall, precision)
            })
        
        return metrics
    
    def save_detailed_results(self, model_name, metrics, y_true, y_pred, inference_time=None):
        """Save detailed results to CSV and JSON in organized folders"""
        
        # Add metadata
        results = {
            'model_name': model_name,
            'evaluation_timestamp': self.evaluation_timestamp,
            'test_samples': len(y_true),
            'ecg_samples': int(np.sum(y_true == 0)),
            'eeg_samples': int(np.sum(y_true == 1)),
            'inference_time_ms': inference_time * 1000 if inference_time else None,
            **metrics
        }
        
        # Save individual model metrics to metrics folder
        individual_csv_path = self.results_structure['metrics'] / f'{model_name}_metrics_{self.evaluation_timestamp}.csv'
        df_individual = pd.DataFrame([results])
        df_individual.to_csv(individual_csv_path, index=False)
        
        # Save to summary CSV in comparisons folder (append mode for comparison)
        summary_csv_path = self.results_structure['comparisons'] / f'evaluation_summary_{self.evaluation_timestamp}.csv'
        
        if summary_csv_path.exists():
            df_existing = pd.read_csv(summary_csv_path)
            df_combined = pd.concat([df_existing, df_individual], ignore_index=True)
            df_combined.to_csv(summary_csv_path, index=False)
        else:
            df_individual.to_csv(summary_csv_path, index=False)
        
        # Save detailed JSON report to detailed_reports folder
        detailed_json_path = self.results_structure['detailed_reports'] / f'{model_name}_detailed_{self.evaluation_timestamp}.json'
        import json
        with open(detailed_json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"   💾 Metrics saved: {individual_csv_path.relative_to(self.results_dir)}")
        print(f"   💾 Summary updated: {summary_csv_path.relative_to(self.results_dir)}")
        print(f"   💾 Detailed report: {detailed_json_path.relative_to(self.results_dir)}")
        
        # Store in instance for summary
        self.all_results[model_name] = results
    
    def print_evaluation_summary(self, model_name, metrics, y_true, y_pred):
        """Print comprehensive evaluation summary"""
        print(f"\n📊 EVALUATION SUMMARY: {model_name.upper()}")
        print("=" * 60)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"Confusion Matrix:")
        print(f"              Predicted")
        print(f"           ECG    EEG")
        print(f"Actual ECG  {cm[0,0]:3d}    {cm[0,1]:3d}")
        print(f"       EEG  {cm[1,0]:3d}    {cm[1,1]:3d}")
        
        # Metrics
        print(f"\nPerformance Metrics:")
        print(f"   Accuracy:      {metrics['accuracy']:.4f}")
        print(f"   Precision:     {metrics['precision']:.4f}")
        print(f"   Recall:        {metrics['recall']:.4f}")
        print(f"   F1-Score:      {metrics['f1_score']:.4f}")
        print(f"   MCC:           {metrics['mcc']:.4f}")
        print(f"   Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"   ROC AUC:       {metrics['roc_auc']:.4f}")
            print(f"   PR AUC:        {metrics['pr_auc']:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["ECG", "EEG"]))
    
    def evaluate_svm(self, model_path, X_test, y_test, metadata):
        """Evaluate SVM model with proper data preparation"""
        print("\n🔍 Evaluating SVM...")
        monitor_memory("before SVM evaluation")
        
        try:
            # Load SVM model
            svm_model = joblib.load(model_path)
            print(f"   ✅ SVM model loaded from {model_path}")
            
            # Prepare data (SVM needs flattened input)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            print(f"   📊 Input shape: {X_test_flat.shape}")
            
            # Predict with timing
            start_time = time.time()
            y_pred = svm_model.predict(X_test_flat)
            y_probs = svm_model.predict_proba(X_test_flat)[:, 1]  # Probability of class 1 (EEG)
            inference_time = time.time() - start_time
            
            print(f"   ⏱️  Inference time: {inference_time*1000:.1f}ms for {len(X_test)} samples")
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_probs)
            
            # Generate visualizations
            self.plot_confusion_matrix(y_test, y_pred, "SVM")
            self.plot_roc_curve(y_test, y_probs, "SVM")
            
            # Print and save results
            self.print_evaluation_summary("SVM", metrics, y_test, y_pred)
            self.save_detailed_results("SVM", metrics, y_test, y_pred, inference_time)
            
        except Exception as e:
            print(f"   ❌ SVM evaluation failed: {str(e)}")
        finally:
            monitor_memory("after SVM evaluation")
    
    def evaluate_keras_model(self, model_path, X_test, y_test, model_name, metadata):
        """Evaluate Keras model with proper data preparation and memory management"""
        print(f"\n🔍 Evaluating {model_name}...")
        monitor_memory(f"before {model_name} evaluation")
        
        try:
            # Load model
            model = load_model(model_path)
            print(f"   ✅ Model loaded from {model_path}")
            print(f"   📊 Model parameters: {model.count_params():,}")
            
            # Prepare data with correct shape
            X_test_model, input_shape = self.prepare_model_data(X_test, model_name, metadata)
            print(f"   📊 Input shape: {X_test_model.shape}")
            
            # Verify model input compatibility
            expected_shape = model.input_shape[1:]  # Remove batch dimension
            if X_test_model.shape[1:] != expected_shape:
                print(f"   ⚠️  Warning: Input shape mismatch!")
                print(f"      Model expects: {expected_shape}")
                print(f"      Data provides: {X_test_model.shape[1:]}")
            
            # Predict with timing
            start_time = time.time()
            y_probs = model.predict(X_test_model, verbose=0)
            inference_time = time.time() - start_time
            
            print(f"   ⏱️  Inference time: {inference_time*1000:.1f}ms for {len(X_test)} samples")
            
            # Handle different output shapes
            if model.output_shape[-1] == 1:
                # Binary classification with sigmoid output
                y_pred = (y_probs > 0.5).astype(int).flatten()
                y_probs_class1 = y_probs.flatten()
            else:
                # Multi-class with softmax output
                y_pred = np.argmax(y_probs, axis=1)
                y_probs_class1 = y_probs[:, 1]  # Probability of class 1 (EEG)
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_probs_class1)
            
            # Generate visualizations
            self.plot_confusion_matrix(y_test, y_pred, model_name)
            self.plot_roc_curve(y_test, y_probs_class1, model_name)
            
            # Print and save results
            self.print_evaluation_summary(model_name, metrics, y_test, y_pred)
            self.save_detailed_results(model_name, metrics, y_test, y_pred, inference_time)
            
            # Clean up model
            del model
            
        except Exception as e:
            print(f"   ❌ {model_name} evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Memory cleanup
            cleanup_memory()
            monitor_memory(f"after {model_name} evaluation")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report of all evaluated models"""
        if not self.all_results:
            print("⚠️  No evaluation results to compare")
            return
        
        print(f"\n🏆 MODEL COMPARISON REPORT")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in self.all_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC AUC': results.get('roc_auc', 'N/A'),
                'PR AUC': results.get('pr_auc', 'N/A'),
                'MCC': results['mcc'],
                'Inference (ms)': results.get('inference_time_ms', 'N/A')
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sort by F1-Score (good overall metric)
        if len(df_comparison) > 1:
            df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Save comparison to comparisons folder
        comparison_path = self.results_structure['comparisons'] / f'model_comparison_{self.evaluation_timestamp}.csv'
        df_comparison.to_csv(comparison_path, index=False)
        print(f"\n💾 Comparison saved: {comparison_path.relative_to(self.results_dir)}")
        
        # Also save a human-readable summary report
        summary_report_path = self.results_structure['comparisons'] / f'performance_summary_{self.evaluation_timestamp}.txt'
        with open(summary_report_path, 'w') as f:
            f.write(f"MODEL PERFORMANCE SUMMARY\n")
            f.write(f"Evaluation Date: {self.evaluation_timestamp}\n")
            f.write(f"Total Models Evaluated: {len(df_comparison)}\n")
            f.write(f"{'='*60}\n\n")
            f.write(df_comparison.to_string(index=False, float_format='%.4f'))
            f.write(f"\n\n{'='*60}\n")
            f.write(f"PERFORMANCE RANKINGS:\n")
            if len(df_comparison) > 1:
                f.write(f"🥇 Best Overall (F1): {df_comparison.iloc[0]['Model']}\n")
                f.write(f"🎯 Best Accuracy: {df_comparison.loc[df_comparison['Accuracy'].idxmax(), 'Model']}\n")
                if 'ROC AUC' in df_comparison.columns and df_comparison['ROC AUC'].dtype != object:
                    f.write(f"📈 Best ROC AUC: {df_comparison.loc[df_comparison['ROC AUC'].idxmax(), 'Model']}\n")
                if 'Inference (ms)' in df_comparison.columns and df_comparison['Inference (ms)'].dtype != object:
                    fastest_model = df_comparison.loc[df_comparison['Inference (ms)'].idxmin(), 'Model']
                    f.write(f"⚡ Fastest: {fastest_model}\n")
        
        print(f"💾 Summary report: {summary_report_path.relative_to(self.results_dir)}")
        
        # Identify best models
        if len(df_comparison) > 1:
            print(f"\n🏅 PERFORMANCE RANKINGS:")
            print(f"   🥇 Best Overall (F1): {df_comparison.iloc[0]['Model']}")
            print(f"   🎯 Best Accuracy: {df_comparison.loc[df_comparison['Accuracy'].idxmax(), 'Model']}")
            
            if 'ROC AUC' in df_comparison.columns and df_comparison['ROC AUC'].dtype != object:
                print(f"   📈 Best ROC AUC: {df_comparison.loc[df_comparison['ROC AUC'].idxmax(), 'Model']}")
            
            if 'Inference (ms)' in df_comparison.columns and df_comparison['Inference (ms)'].dtype != object:
                fastest_model = df_comparison.loc[df_comparison['Inference (ms)'].idxmin(), 'Model']
                print(f"   ⚡ Fastest: {fastest_model}")

def main():
    """Main evaluation pipeline"""
    print("🚀 MODERN MODEL EVALUATION PIPELINE")
    print("=" * 70)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Monitor initial memory
    monitor_memory("initial")
    
    try:
        # Load cached dataset
        print("\n📦 Loading cached dataset...")
        X_train, X_test, y_train, y_test, metadata = evaluator.load_cached_dataset()
        
        # Find available models
        print("\n🔍 Discovering trained models...")
        available_models = evaluator.find_available_models()
        
        if not available_models:
            print("❌ No trained models found! Please run training first:")
            print("   python train.py all")
            return
        
        print(f"\n🎯 Evaluating {len(available_models)} models...")
        
        # Evaluate each model
        for model_name, model_path in available_models.items():
            print(f"\n{'='*20} {model_name.upper()} {'='*20}")
            
            try:
                if model_name == 'svm':
                    evaluator.evaluate_svm(model_path, X_test, y_test, metadata)
                else:
                    evaluator.evaluate_keras_model(model_path, X_test, y_test, model_name, metadata)
                    
            except Exception as e:
                print(f"❌ Failed to evaluate {model_name}: {str(e)}")
                continue
        
        # Generate comparison report
        print("\n" + "="*70)
        evaluator.generate_comparison_report()
        
        print(f"\n✅ EVALUATION COMPLETE!")
        print(f"📁 Results organized in: {evaluator.results_dir}")
        print(f"   📊 Confusion matrices: {RESULTS_STRUCTURE['confusion_matrices']}/")
        print(f"   📈 ROC/PR curves: {RESULTS_STRUCTURE['roc_curves']}/") 
        print(f"   📋 Individual metrics: {RESULTS_STRUCTURE['metrics']}/")
        print(f"   🏆 Comparisons: {RESULTS_STRUCTURE['comparisons']}/")
        print(f"   📄 Detailed reports: {RESULTS_STRUCTURE['detailed_reports']}/")
        print(f"🕒 Timestamp: {evaluator.evaluation_timestamp}")
        
    except Exception as e:
        print(f"❌ Evaluation pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        cleanup_memory()
        monitor_memory("final")

if __name__ == '__main__':
    # Handle command line arguments for advanced usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained ECG vs EEG classification models')
    parser.add_argument('--models', nargs='+', 
                       choices=['svm', 'simple_cnn', 'cnn_lstm', 'mlp', 'tcn', 'dual_branch'],
                       help='Specific models to evaluate (default: all available)')
    parser.add_argument('--force-reload', action='store_true',
                       help='Force reload dataset (ignore cache)')
    parser.add_argument('--results-dir', default=RESULTS_DIR,
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Update global results directory if specified
    if args.results_dir != RESULTS_DIR:
        RESULTS_DIR = args.results_dir
        print(f"📁 Using custom results directory: {RESULTS_DIR}")
    
    main()