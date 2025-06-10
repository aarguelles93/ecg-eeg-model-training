import os
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib
import json
from datetime import datetime

# UPDATED: Import from training_utils instead of direct TF imports
from training_utils import create_optimizer, cleanup_memory

from models import (
    build_simple_cnn, build_cnn_lstm, build_mlp, build_tcn, 
    build_dual_branch_cnn, build_svm_model
)
from config import CONFIG

class LearningCurveAnalyzer:
    def __init__(self, output_dir="learning_curves", random_state=42):
        self.output_dir = output_dir
        self.random_state = random_state
        os.makedirs(output_dir, exist_ok=True)
        
        # Default sample sizes (as fractions of total dataset)
        self.default_sample_fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]
        
        # Early stopping criteria
        self.early_stop_config = {
            'patience_samples': 3,  # Stop if no improvement for 3 consecutive sample sizes
            'min_improvement': 0.005,  # Minimum improvement threshold (0.5%)
            'min_samples': 0.1,  # Always train on at least 10% of data
            'max_time_per_size': 3600  # Max 1 hour per sample size
        }
    
    def prepare_model_data(self, X_train, X_test, model_name):
        """Prepare data with correct shape for different model types"""
        if model_name == 'mlp':
            X_train_model = X_train.reshape(X_train.shape[0], -1)
            X_test_model = X_test.reshape(X_test.shape[0], -1)
            input_shape = (X_train_model.shape[1],)
        else:
            if len(X_train.shape) == 2:
                X_train_model = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test_model = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            else:
                X_train_model = X_train
                X_test_model = X_test
            input_shape = X_train_model.shape[1:]
        
        return X_train_model, X_test_model, input_shape
    
    def get_model_builder(self, model_name):
        """Get the appropriate model builder function"""
        builders = {
            'simple_cnn': build_simple_cnn,
            'cnn_lstm': build_cnn_lstm,
            'mlp': build_mlp,
            'tcn': build_tcn,
            'dual_branch': build_dual_branch_cnn,
            'svm': build_svm_model
        }
        return builders.get(model_name)
    
    def train_model_subset(self, model_name, X_train, y_train, X_val, y_val, sample_size_fraction):
        """Train a model on a subset of data and return metrics - FIXED VERSION"""
        
        # Sample training data
        n_samples = int(len(X_train) * sample_size_fraction)
        if n_samples < 50:  # Minimum viable sample size
            n_samples = min(50, len(X_train))
        
        # Stratified sampling to maintain class balance
        from sklearn.model_selection import train_test_split
        X_subset, _, y_subset, _ = train_test_split(
            X_train, y_train, 
            train_size=n_samples,
            stratify=y_train,
            random_state=self.random_state
        )

        print(f"   ⚙️  Using config for {model_name}: {CONFIG[model_name]}")
        
        start_time = time.time()
        
        try:
            if model_name == 'svm':
                X_subset_flat = X_subset.reshape(X_subset.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                
                model = build_svm_model(y_subset, config=CONFIG['svm'])
                model.fit(X_subset_flat, y_subset)
                
                train_pred = model.predict(X_subset_flat)
                val_pred = model.predict(X_val_flat)
                
                train_acc = accuracy_score(y_subset, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                
            else:
                # Neural network training
                import tensorflow as tf
                from tensorflow.keras.callbacks import EarlyStopping
                
                # Clear any existing session first
                tf.keras.backend.clear_session()
                
                X_subset_model, X_val_model, input_shape = self.prepare_model_data(
                    X_subset, X_val, model_name
                )
                
                builder = self.get_model_builder(model_name)
                if builder is None:
                    raise ValueError(f"No model builder found for model_name '{model_name}'. Check spelling or add to builders.")
                model = builder(input_shape=input_shape, num_classes=2, config=CONFIG[model_name])
                
                # FIX: Use centralized optimizer creation
                learning_rate = CONFIG[model_name].get('learning_rate', 1e-4)
                optimizer = create_optimizer(learning_rate=learning_rate)
                
                model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Early stopping for efficiency
                early_stop = EarlyStopping(
                    monitor='val_accuracy', 
                    patience=3,  # Reduced patience for learning curve
                    restore_best_weights=True,
                    verbose=0
                )
                
                # Reduce epochs and batch size for learning curve analysis
                max_epochs = 15
                batch_size = min(32, CONFIG[model_name].get('batch_size', 64))  # Smaller batches
                
                history = model.fit(
                    X_subset_model, y_subset,
                    validation_data=(X_val_model, y_val),
                    epochs=max_epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Get final accuracies
                train_acc = history.history['accuracy'][-1]
                val_acc = history.history['val_accuracy'][-1]
                
                # Clean up model immediately
                del model
                tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"   ❌ Error training {model_name}: {str(e)}")
            # Return default values on error
            return {
                'sample_size': n_samples,
                'sample_fraction': sample_size_fraction,
                'train_accuracy': 0.5,
                'val_accuracy': 0.5,
                'training_time': time.time() - start_time,
                'samples_per_second': 0
            }
        
        finally:
            # ENHANCED: Explicit cleanup of all TF/Keras objects
            try:
                del model
            except: pass
            try:
                del optimizer
            except: pass
            try:
                del history
            except: pass
            try:
                del early_stop
            except: pass

            import tensorflow as tf
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
            import gc
            gc.collect()
            gc.collect()
        
        training_time = time.time() - start_time
        
        return {
            'sample_size': n_samples,
            'sample_fraction': sample_size_fraction,
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'training_time': float(training_time),
            'samples_per_second': float(n_samples / training_time if training_time > 0 else 0)
        }


    
    def should_early_stop(self, results, current_idx):
        """Determine if we should stop the learning curve early"""
        if current_idx < 2:  # Need at least 3 points
            return False
        
        # Check if we've met minimum sample requirement
        current_fraction = self.default_sample_fractions[current_idx]
        if current_fraction < self.early_stop_config['min_samples']:
            return False
        
        # Look at last few validation accuracies
        recent_val_accs = []
        for r in results[-self.early_stop_config['patience_samples']:]:
            if 'val_accuracy_mean' in r:
                recent_val_accs.append(r['val_accuracy_mean'])
            elif 'val_accuracy' in r:
                recent_val_accs.append(r['val_accuracy'])
            else:
                print(f"⚠️ Missing validation accuracy in result: {r}")
        
        if len(recent_val_accs) < self.early_stop_config['patience_samples']:
            return False
        
        # Check if improvement has plateaued
        max_recent = max(recent_val_accs)
        improvement = max_recent - recent_val_accs[0]
        
        if improvement < self.early_stop_config['min_improvement']:
            print(f"   🛑 Early stopping: Improvement {improvement:.4f} < threshold {self.early_stop_config['min_improvement']}")
            return True
        
        return False
    
    def analyze_single_model(self, model_name, X_train, y_train, X_test, y_test, 
                       sample_fractions=None, n_folds=3):
        """Analyze learning curve for a single model - FIXED VERSION"""
        
        print(f"\n🔍 Analyzing learning curve for {model_name.upper()}")
        print("=" * 60)
        
        if sample_fractions is None:
            sample_fractions = self.default_sample_fractions
        
        # Prepare data
        X_train_model, X_test_model, input_shape = self.prepare_model_data(X_train, X_test, model_name)
        
        # Results storage
        all_results = []
        
        # Stratified K-Fold for robust evaluation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        for i, sample_fraction in enumerate(sample_fractions):
            print(f"\n📊 Training on {sample_fraction*100:.1f}% of data...")
            
            fold_results = []
            fold_start_time = time.time()
            
            # Memory check before this sample size
            import psutil
            mem_before = psutil.Process().memory_info().rss / 1024**3
            print(f"   💾 Memory before: {mem_before:.2f}GB")
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_model, y_train)):
                X_tr, X_val = X_train_model[train_idx], X_train_model[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                fold_result = self.train_model_subset(
                    model_name, X_tr, y_tr, X_val, y_val, sample_fraction
                )
                fold_results.append(fold_result)
                
                try:
                    print(f"   Fold {fold+1}: Train={fold_result['train_accuracy']:.3f}, "
                          f"Val={fold_result['val_accuracy']:.3f}, "
                          f"Time={fold_result['training_time']:.1f}s")
                except KeyError as e:
                    print(f"   ⚠️ KeyError in fold result: {e}")
                    print(f"   🔍 Fold {fold+1} raw result: {fold_result}")
                
                cleanup_memory()
                
                if time.time() - fold_start_time > self.early_stop_config['max_time_per_size']:
                    print(f"   ⏰ Time limit reached for sample size {sample_fraction}")
                    break
            
            mem_after = psutil.Process().memory_info().rss / 1024**3
            print(f"   💾 Memory after: {mem_after:.2f}GB (Δ{mem_after - mem_before:+.2f}GB)")
            
            # Filter out invalid or incomplete results
            valid_fold_results = [
                r for r in fold_results 
                if r and isinstance(r, dict) and 
                all(k in r for k in ['train_accuracy', 'val_accuracy', 'training_time', 'samples_per_second'])
            ]
            
            if not valid_fold_results:
                print(f"❌ No valid fold results for {model_name} at {sample_fraction*100:.1f}% sample size")
                continue

            try:
                avg_result = {
                    'sample_fraction': sample_fraction,
                    'sample_size': int(len(X_train) * sample_fraction),
                    'train_accuracy_mean': np.mean([r['train_accuracy'] for r in valid_fold_results]),
                    'train_accuracy_std': np.std([r['train_accuracy'] for r in valid_fold_results]),
                    'val_accuracy_mean': np.mean([r['val_accuracy'] for r in valid_fold_results]),
                    'val_accuracy_std': np.std([r['val_accuracy'] for r in valid_fold_results]),
                    'training_time_mean': np.mean([r['training_time'] for r in valid_fold_results]),
                    'training_time_std': np.std([r['training_time'] for r in valid_fold_results]),
                    'samples_per_second': np.mean([r['samples_per_second'] for r in valid_fold_results])
                }
                all_results.append(avg_result)

                print(f"   📈 Average: Train={avg_result['train_accuracy_mean']:.3f}±{avg_result['train_accuracy_std']:.3f}, "
                      f"Val={avg_result['val_accuracy_mean']:.3f}±{avg_result['val_accuracy_std']:.3f}")

                if self.should_early_stop(all_results, i):
                    print(f"   🎯 Optimal dataset size found at {sample_fraction*100:.1f}%")
                    break

            except KeyError as e:
                print(f"❌ Error analyzing {model_name}: {e}")
                for idx, fr in enumerate(fold_results):
                    print(f"   🔍 Fold {idx+1} raw result: {fr}")
            
            del fold_results
            gc.collect()
            cleanup_memory()
        
        cleanup_memory()
        print(f"🧹 Memory cleaned up after {model_name}")
        
        return all_results

    
    def plot_learning_curves(self, results_dict, save_plots=True):
        """Create comprehensive learning curve visualizations"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Learning Curve Analysis - ECG vs EEG Classification', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy vs Dataset Size
        ax1 = axes[0, 0]
        for model_name, results in results_dict.items():
            sample_sizes = [r['sample_size'] for r in results]
            train_accs = [r['train_accuracy_mean'] for r in results]
            val_accs = [r['val_accuracy_mean'] for r in results]
            val_stds = [r['val_accuracy_std'] for r in results]
            
            ax1.plot(sample_sizes, train_accs, '--', alpha=0.7, label=f'{model_name} (train)')
            ax1.errorbar(sample_sizes, val_accs, yerr=val_stds, 
                        marker='o', label=f'{model_name} (val)', capsize=3)
        
        ax1.set_xlabel('Dataset Size (samples)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Dataset Size')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Training Time vs Dataset Size
        ax2 = axes[0, 1]
        for model_name, results in results_dict.items():
            sample_sizes = [r['sample_size'] for r in results]
            train_times = [r['training_time_mean'] for r in results]
            time_stds = [r['training_time_std'] for r in results]
            
            ax2.errorbar(sample_sizes, train_times, yerr=time_stds,
                        marker='s', label=model_name, capsize=3)
        
        ax2.set_xlabel('Dataset Size (samples)')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time vs Dataset Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Plot 3: Efficiency (Samples per Second)
        ax3 = axes[1, 0]
        for model_name, results in results_dict.items():
            sample_sizes = [r['sample_size'] for r in results]
            efficiency = [r['samples_per_second'] for r in results]
            
            ax3.plot(sample_sizes, efficiency, marker='^', label=model_name)
        
        ax3.set_xlabel('Dataset Size (samples)')
        ax3.set_ylabel('Training Efficiency (samples/sec)')
        ax3.set_title('Training Efficiency vs Dataset Size')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Plot 4: Overfitting Analysis (Train-Val Gap)
        ax4 = axes[1, 1]
        for model_name, results in results_dict.items():
            sample_sizes = [r['sample_size'] for r in results]
            overfitting_gap = [r['train_accuracy_mean'] - r['val_accuracy_mean'] for r in results]
            
            ax4.plot(sample_sizes, overfitting_gap, marker='d', label=model_name)
        
        ax4.set_xlabel('Dataset Size (samples)')
        ax4.set_ylabel('Overfitting Gap (Train - Val Accuracy)')
        ax4.set_title('Overfitting Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xscale('log')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.output_dir, f'learning_curves_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Learning curves saved to: {plot_path}")
        
        plt.show()
    
    def generate_recommendations(self, results_dict):
        """Generate recommendations based on learning curve analysis"""
        
        print("\n🎯 LEARNING CURVE ANALYSIS RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = {}
        
        for model_name, results in results_dict.items():
            print(f"\n🔍 {model_name.upper()} Analysis:")
            
            # Find optimal dataset size (highest val accuracy with reasonable efficiency)
            best_val_acc = max(r['val_accuracy_mean'] for r in results)
            best_result = next(r for r in results if r['val_accuracy_mean'] == best_val_acc)
            
            # Find efficiency sweet spot (good accuracy with reasonable time)
            efficient_results = [r for r in results if r['val_accuracy_mean'] >= best_val_acc * 0.95]
            most_efficient = min(efficient_results, key=lambda x: x['training_time_mean'])
            
            # Overfitting analysis
            final_result = results[-1]
            overfitting_gap = final_result['train_accuracy_mean'] - final_result['val_accuracy_mean']
            
            recommendations[model_name] = {
                'optimal_samples': best_result['sample_size'],
                'optimal_fraction': best_result['sample_fraction'],
                'best_val_accuracy': best_val_acc,
                'efficient_samples': most_efficient['sample_size'],
                'efficient_fraction': most_efficient['sample_fraction'],
                'overfitting_severity': 'High' if overfitting_gap > 0.1 else 'Moderate' if overfitting_gap > 0.05 else 'Low',
                'recommended_action': ''
            }
            
            print(f"   🎯 Optimal dataset size: {best_result['sample_size']:,} samples "
                  f"({best_result['sample_fraction']*100:.1f}%) - Accuracy: {best_val_acc:.3f}")
            print(f"   ⚡ Efficient dataset size: {most_efficient['sample_size']:,} samples "
                  f"({most_efficient['sample_fraction']*100:.1f}%) - Time: {most_efficient['training_time_mean']:.1f}s")
            print(f"   🎭 Overfitting level: {recommendations[model_name]['overfitting_severity']} "
                  f"(gap: {overfitting_gap:.3f})")
            
            # Generate specific recommendations
            if best_result['sample_fraction'] < 0.3:
                action = f"✅ EXCELLENT! Model performs well with only {best_result['sample_fraction']*100:.1f}% of data"
            elif best_result['sample_fraction'] < 0.7:
                action = f"👍 GOOD: Use {best_result['sample_fraction']*100:.1f}% of data for optimal results"
            else:
                action = f"⚠️  REQUIRES LARGE DATASET: Needs {best_result['sample_fraction']*100:.1f}% of data"
            
            recommendations[model_name]['recommended_action'] = action
            print(f"   💡 Recommendation: {action}")
        
        # Overall recommendations
        print(f"\n🏆 OVERALL RECOMMENDATIONS:")
        best_models = sorted(recommendations.items(), 
                           key=lambda x: x[1]['best_val_accuracy'], reverse=True)[:3]
        
        print(f"   🥇 Top performing models:")
        for i, (model, rec) in enumerate(best_models, 1):
            print(f"      {i}. {model}: {rec['best_val_accuracy']:.3f} accuracy with "
                  f"{rec['optimal_samples']:,} samples")
        
        most_efficient = min(recommendations.items(), 
                           key=lambda x: x[1]['efficient_samples'])
        print(f"   ⚡ Most efficient model: {most_efficient[0]} - "
              f"Good performance with only {most_efficient[1]['efficient_samples']:,} samples")
        
        return recommendations
    
    def save_results(self, results_dict, recommendations=None):
        """Save results to files for later analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = os.path.join(self.output_dir, f'learning_curve_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"💾 Detailed results saved to: {results_path}")
        
        # Save summary CSV
        summary_data = []
        for model_name, results in results_dict.items():
            for result in results:
                summary_data.append({
                    'model': model_name,
                    **result
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, f'learning_curve_summary_{timestamp}.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 Summary CSV saved to: {summary_path}")
        
        # Save recommendations
        if recommendations:
            rec_path = os.path.join(self.output_dir, f'recommendations_{timestamp}.json')
            with open(rec_path, 'w') as f:
                json.dump(recommendations, f, indent=2)
            print(f"💡 Recommendations saved to: {rec_path}")

def run_learning_curve_analysis(X_train, y_train, X_test, y_test, 
                               models_to_analyze=None, 
                               sample_fractions=None,
                               n_folds=3,
                               output_dir="learning_curves"):
    """
    Main function to run learning curve analysis
    
    Args:
        X_train, y_train, X_test, y_test: Dataset splits
        models_to_analyze: List of model names to analyze (default: all models)
        sample_fractions: List of sample fractions to test
        n_folds: Number of CV folds
        output_dir: Output directory for results
    """
    
    if models_to_analyze is None:
        # Focus on faster models for learning curve analysis
        models_to_analyze = ['mlp', 'simple_cnn', 'svm']
    
    analyzer = LearningCurveAnalyzer(output_dir=output_dir)
    
    print("🚀 LEARNING CURVE ANALYSIS STARTING")
    print("=" * 60)
    print(f"📊 Dataset size: {len(X_train):,} training samples")
    print(f"🔍 Models to analyze: {', '.join(models_to_analyze)}")
    print(f"📈 Sample fractions: {sample_fractions or analyzer.default_sample_fractions}")
    print(f"🔄 Cross-validation folds: {n_folds}")
    
    results_dict = {}
    
    for model_name in models_to_analyze:
        try:
            results = analyzer.analyze_single_model(
                model_name, X_train, y_train, X_test, y_test,
                sample_fractions=sample_fractions,
                n_folds=n_folds
            )
            results_dict[model_name] = results
            
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {str(e)}")
            continue
        finally:
            # UPDATED: Use centralized cleanup
            cleanup_memory()
    
    if results_dict:
        # Generate visualizations and recommendations
        analyzer.plot_learning_curves(results_dict)
        recommendations = analyzer.generate_recommendations(results_dict)
        analyzer.save_results(results_dict, recommendations)
        
        return results_dict, recommendations
    else:
        print("❌ No models were successfully analyzed!")
        return None, None

# Example usage function for integration with train.py
def integrate_with_training(X_train, y_train, X_test, y_test):
    """Example of how to integrate learning curve analysis with existing training"""
    
    print("🎯 Running quick learning curve analysis before full training...")
    
    # Run analysis on subset of models first
    results, recommendations = run_learning_curve_analysis(
        X_train, y_train, X_test, y_test,
        models_to_analyze=['mlp', 'simple_cnn'],  # Fast models for initial analysis
        sample_fractions=[0.05, 0.1, 0.2, 0.5],  # Reduced fractions for speed
        n_folds=2  # Reduced folds for speed
    )
    
    if recommendations:
        print("\n💡 Based on learning curve analysis, consider:")
        for model, rec in recommendations.items():
            if rec['efficient_fraction'] < 0.5:
                print(f"   • {model}: Can achieve good results with {rec['efficient_samples']:,} samples")
                print(f"     This could reduce training time significantly!")
    
    return results, recommendations