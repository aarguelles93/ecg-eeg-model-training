import env_setup
import os
import gc
import warnings
import joblib
import numpy as np
import csv
import gc
import time
import psutil
from tqdm import tqdm

# UPDATED: Import from training_utils instead of direct TF imports
from training_utils import create_optimizer, cleanup_memory, monitor_memory, compute_class_weights

from data_loader import prepare_dataset
from models import (
    build_simple_cnn, build_cnn_lstm, build_svm_model, build_tcn,
    build_mlp, build_dual_branch_cnn
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

# ============================================================================

# GPU Configuration
from gpu_config import gpu_friendly_config, get_gpu_optimized_config, monitor_gpu_memory, cleanup_tensorflow_memory

import tensorflow as tf
# _ = tf.keras.layers.LSTM(1)  # Trigger TF initialization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Configure GPU BEFORE any TensorFlow operations
print("🔧 Setting up GPU configuration...")
GPU_AVAILABLE = gpu_friendly_config()

if GPU_AVAILABLE:
    print("✅ GPU configuration successful - using optimized settings")
    # Import original config
    from config import CONFIG
    # Get GPU-optimized config
    GPU_CONFIG = get_gpu_optimized_config()
    
    # Create a copy of CONFIG to avoid mutation issues
    RUNTIME_CONFIG = CONFIG.copy()
    for model_name in RUNTIME_CONFIG:
        if isinstance(RUNTIME_CONFIG[model_name], dict):
            RUNTIME_CONFIG[model_name] = RUNTIME_CONFIG[model_name].copy()
    
    # Merge GPU optimizations with original config
    for model_name in ['simple_cnn', 'cnn_lstm', 'mlp', 'tcn', 'dual_branch']:
        if model_name in GPU_CONFIG:
            RUNTIME_CONFIG[model_name].update(GPU_CONFIG[model_name])
    
    # Reduce global settings for GPU efficiency
    # RUNTIME_CONFIG['global']['epochs'] = 30  # Reduced from 50
    # RUNTIME_CONFIG['global']['n_splits'] = 3  # Reduced from 5
    
else:
    print("⚠️ GPU not available - using original CPU settings")
    from config import CONFIG
    RUNTIME_CONFIG = CONFIG

# ============================================================================
# MEMORY MANAGEMENT UTILITIES (UPDATED to use training_utils)
# ============================================================================

def get_memory_usage():
    """Get current memory usage statistics"""
    process = psutil.Process(os.getpid())
    cpu_memory_gb = process.memory_info().rss / 1024**3
    
    gpu_memory_gb = 0
    if GPU_AVAILABLE:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_gb = info.used / 1024**3
        except ImportError:
            gpu_memory_gb = 0
    
    return cpu_memory_gb, gpu_memory_gb

def cleanup_between_models():
    """Aggressive memory cleanup between model training"""
    print("🧹 Performing memory cleanup...")
    
    # Use centralized cleanup from training_utils
    cleanup_memory()
    
    # Memory status after cleanup
    cpu_mem, gpu_mem = get_memory_usage()
    print(f"🧹 Cleanup complete - CPU: {cpu_mem:.2f}GB, GPU: {gpu_mem:.2f}GB")

# UPDATED: Use training_utils version instead of local function
def monitor_memory_usage(label=""):
    """Monitor and log current memory usage"""
    return monitor_memory(label)

# ============================================================================
# PROGRESS TRACKING UTILITIES
# ============================================================================

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for detailed training progress with tqdm"""
    
    def __init__(self, model_name, position=1):
        super().__init__()
        self.model_name = model_name
        self.position = position
        self.progress_bar = None
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.progress_bar = tqdm(
            total=self.params['epochs'], 
            desc=f"📈 {self.model_name}",
            position=self.position,
            leave=True,
            unit="epoch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
    
    def on_epoch_end(self, epoch, logs=None):
        # Get learning rate
        try:
            current_lr = float(self.model.optimizer.learning_rate.numpy())
        except:
            current_lr = 0
        
        # Build progress postfix
        postfix = {
            'Loss': f"{logs.get('loss', 0):.4f}",
            'Acc': f"{logs.get('accuracy', 0):.3f}",
            'Val_Acc': f"{logs.get('val_accuracy', 0):.3f}",
            'LR': f"{current_lr:.1e}",
        }
        
        # Add memory info if available
        cpu_mem, gpu_mem = get_memory_usage()
        if gpu_mem > 0:
            postfix['GPU'] = f"{gpu_mem:.1f}GB"
        
        self.progress_bar.set_postfix(postfix)
        self.progress_bar.update(1)
    
    def on_train_end(self, logs=None):
        if self.progress_bar:
            training_time = time.time() - self.start_time
            self.progress_bar.set_description(f"✅ {self.model_name} ({training_time:.1f}s)")
            self.progress_bar.close()

def estimate_remaining_time(current_index, total_models, start_time):
    """Estimate remaining training time"""
    if current_index == 0:
        return "Unknown"
    
    elapsed = time.time() - start_time
    avg_time_per_model = elapsed / (current_index + 1)
    remaining_models = total_models - current_index - 1
    remaining_seconds = remaining_models * avg_time_per_model
    
    if remaining_seconds < 60:
        return f"{remaining_seconds:.0f}s"
    elif remaining_seconds < 3600:
        return f"{remaining_seconds/60:.1f}m"
    else:
        return f"{remaining_seconds/3600:.1f}h"

# ============================================================================
# SMART EARLY STOPPING
# ============================================================================

class AdaptiveEarlyStopping(tf.keras.callbacks.Callback):
    """
    Sophisticated early stopping with multiple criteria and noise tolerance
    """
    
    def __init__(self, 
                 monitor='val_accuracy',
                 min_delta=0.001,
                 patience=10,
                 baseline_patience=5,      # Patience for reaching baseline
                 noise_tolerance=0.002,    # Tolerance for noisy improvements 
                 min_epochs=20,           # Minimum epochs before stopping
                 factor_patience=True,     # Scale patience by model complexity
                 restore_best_weights=True):
        
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.baseline_patience = baseline_patience
        self.noise_tolerance = noise_tolerance
        self.min_epochs = min_epochs
        self.factor_patience = factor_patience
        self.restore_best_weights = restore_best_weights
        
        # Adaptive patience based on model complexity
        self.patience = patience
        self.original_patience = patience
        
        # Tracking variables
        self.best_weights = None
        self.best_epoch = 0
        self.best_value = None
        self.wait = 0
        self.baseline_wait = 0
        self.values_history = []
    
    def on_train_begin(self, logs=None):
        # Now self.model is available - adapt patience based on model complexity
        if self.factor_patience and hasattr(self.model, 'count_params'):
            try:
                total_params = self.model.count_params()
                if total_params > 1_000_000:  # Large model
                    self.patience = int(self.original_patience * 1.5)
                    print(f"🧠 Large model ({total_params:,} params) - patience increased to {self.patience}")
                elif total_params < 100_000:  # Small model  
                    self.patience = max(5, int(self.original_patience * 0.7))
                    print(f"🧠 Small model ({total_params:,} params) - patience reduced to {self.patience}")
            except Exception as e:
                print(f"⚠️  Could not count model parameters: {e}")
                # Use original patience as fallback
                pass
        
        self.best_value = -np.inf if 'acc' in self.monitor else np.inf
        self.wait = 0
        self.baseline_wait = 0
        self.values_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Safely get the monitored value
        if logs is None:
            return
            
        current_value = logs.get(self.monitor)
        if current_value is None:
            print(f"⚠️  Warning: '{self.monitor}' not found in logs. Available metrics: {list(logs.keys())}")
            return
        
        self.values_history.append(current_value)
        
        # Skip early epochs
        if epoch < self.min_epochs:
            return
        
        # Check for improvement with noise tolerance
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            self.baseline_wait = 0
            
            if self.restore_best_weights and self.model and hasattr(self.model, 'get_weights'):
                try:
                    self.best_weights = self.model.get_weights()
                except Exception as e:
                    print(f"⚠️  Could not save weights: {e}")
        
        else:
            self.wait += 1
            self.baseline_wait += 1
            
            # Check if we're stuck at baseline (no learning)
            if self._is_stuck_at_baseline() and self.baseline_wait >= self.baseline_patience:
                print(f"🛑 Early stopping: Model stuck at baseline for {self.baseline_wait} epochs")
                if self.model and hasattr(self.model, 'stop_training'):
                    self.model.stop_training = True
                return
            
            # Check for plateau with trend analysis
            if self._is_plateau_with_trend_analysis():
                print(f"🛑 Early stopping: Plateau detected after {self.wait} epochs")
                print(f"   Best {self.monitor}: {self.best_value:.6f} at epoch {self.best_epoch}")
                
                if self.restore_best_weights and self.best_weights and self.model and hasattr(self.model, 'set_weights'):
                    try:
                        print("🔄 Restoring best weights")
                        self.model.set_weights(self.best_weights)
                    except Exception as e:
                        print(f"⚠️  Could not restore weights: {e}")
                
                if self.model and hasattr(self.model, 'stop_training'):
                    self.model.stop_training = True
    
    def _is_improvement(self, current_value):
        """Check if current value is an improvement considering noise tolerance"""
        if 'acc' in self.monitor:  # Higher is better
            improvement = current_value - self.best_value
            return improvement > max(self.min_delta, self.noise_tolerance)
        else:  # Lower is better (loss)
            improvement = self.best_value - current_value  
            return improvement > max(self.min_delta, self.noise_tolerance)
    
    def _is_stuck_at_baseline(self):
        """Detect if model is stuck at random baseline performance"""
        if len(self.values_history) < self.baseline_patience:
            return False
        
        recent_values = self.values_history[-self.baseline_patience:]
        
        # For accuracy: stuck around 0.5 (random)
        if 'acc' in self.monitor:
            baseline_range = (0.45, 0.55)
            return all(baseline_range[0] <= val <= baseline_range[1] for val in recent_values)
        
        # For loss: not decreasing from initial high values
        else:
            initial_loss = self.values_history[0] if self.values_history else float('inf')
            recent_avg = np.mean(recent_values)
            return recent_avg > 0.9 * initial_loss  # Still 90% of initial loss
    
    def _is_plateau_with_trend_analysis(self):
        """Advanced plateau detection using trend analysis"""
        if self.wait < self.patience:
            return False
        
        # Look at recent trend
        if len(self.values_history) >= 10:
            recent_values = np.array(self.values_history[-10:])
            
            # Calculate trend slope
            x = np.arange(len(recent_values))
            slope = np.polyfit(x, recent_values, 1)[0]
            
            # For accuracy: positive slope means improvement
            # For loss: negative slope means improvement
            expected_sign = 1 if 'acc' in self.monitor else -1
            
            # If trend is flat or going wrong direction
            if abs(slope) < 0.001 or np.sign(slope) != expected_sign:
                return True
        
        return True  # Default: stop after patience epochs

# ============================================================================
# CORE TRAINING FUNCTIONS (Updated with new utilities)
# ============================================================================

def choose_validation_strategy(dataset_size, time_budget='medium', force_strategy=None):
    """
    Auto-select validation strategy based on dataset size and constraints
    """
    if force_strategy:
        return force_strategy
    
    if time_budget == 'low' or dataset_size > 50000:
        return 'split'
    elif dataset_size < 5000:
        return 'kfold'  # Small datasets benefit from kfold
    else:
        return 'split'  # Default for medium-large datasets

def save_training_curves(history, model_name, output_dir="results"):
    import datetime
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{model_name}_training_{timestamp}.png")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train Acc')
    plt.plot(history.history.get('val_accuracy', []), label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"📈 Training curves saved to: {filepath}")

def save_roc_pr_curves(y_true, y_probs, model_name, output_dir="results"):
    """Save ROC and Precision-Recall curves"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{model_name}_roc_pr_{timestamp}.png")
    
    # For binary classification, get probabilities for positive class
    if len(y_probs.shape) > 1 and y_probs.shape[1] > 1:
        y_probs_pos = y_probs[:, 1]  # Multi-class: take positive class
    else:
        y_probs_pos = y_probs.flatten()  # Binary: use as-is
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs_pos)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve  
    precision, recall, _ = precision_recall_curve(y_true, y_probs_pos)
    pr_auc = average_precision_score(y_true, y_probs_pos)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'{model_name} - ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='darkred', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax2.axhline(y=0.5, color='navy', linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision') 
    ax2.set_title(f'{model_name} - Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 ROC and PR curves saved to: {filepath}")

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Reusable confusion matrix plotting"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import datetime
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ECG', 'EEG'], yticklabels=['ECG', 'EEG'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} Confusion Matrix")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f'{model_name}_confusion_matrix_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"📉 Confusion matrix saved to: {filepath}")

def log_model_configuration(model_name, config):
    """
    Log the configuration parameters being used for a specific model
    
    Args:
        model_name: Name of the model being trained
        config: Configuration dictionary for the model
    """
    print(f"⚙️  Configuration for {model_name.upper()}:")
    
    # Format configuration nicely
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"   📋 {key}:")
            for sub_key, sub_value in value.items():
                print(f"      {sub_key}: {sub_value}")
        else:
            print(f"   📋 {key}: {value}")
    
    print()  

def log_fold_metrics_to_csv(model_name, fold_scores, output_dir="results"):
    csv_path = os.path.join(output_dir, f"{model_name}_fold_metrics.csv")
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Fold", "Validation Accuracy"])
        for i, score in enumerate(fold_scores):
            writer.writerow([f"Fold {i+1}", f"{score:.4f}"])
        writer.writerow([])
        writer.writerow(["Mean", f"{np.mean(fold_scores):.4f}"])
        writer.writerow(["Std Dev", f"{np.std(fold_scores):.4f}"])

    print(f"📝 Fold metrics saved to: {csv_path}")

def print_evaluation(y_true, y_pred, model_name):
    print(f"\nEvaluation for {model_name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["ECG", "EEG"]))

def train_svm(X_train, y_train, X_test, y_test, output_dir):
    print("\n🔧 Training SVM...")
    monitor_memory_usage("before SVM")
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print(f"🔍 SVM Data Analysis:")
    print(f"   📊 Train shape: {X_train_flat.shape}")
    print(f"   📊 Test shape: {X_test_flat.shape}")
    print(f"   📊 Feature range: [{X_train_flat.min():.6f}, {X_train_flat.max():.6f}]")
    print(f"   📊 Train labels: ECG={np.sum(y_train==0)}, EEG={np.sum(y_train==1)}")
    print(f"   📊 Test labels: ECG={np.sum(y_test==0)}, EEG={np.sum(y_test==1)}")
    
    # Check if data is the projected ECG format
    if X_train_flat.shape[1] == 6016:
        print(f"   ✅ Using projected ECG features (6016 = 32×188)")
    elif X_train_flat.shape[1] == 187:
        print(f"   ⚠️  Using raw ECG features (187) - may cause poor performance!")
    else:
        print(f"   ⚠️  Unexpected feature count: {X_train_flat.shape[1]}")

    # SVM with progress bar
    print("📊 Building SVM model...")
    model = build_svm_model(y_train, config=RUNTIME_CONFIG['svm'])
    
    print("🚀 Training SVM...")
    start_time = time.time()
    model.fit(X_train_flat, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ SVM training completed in {training_time:.1f}s")

    y_pred = model.predict(X_test_flat)
    print_evaluation(y_test, y_pred, "SVM")
    y_probs_svm = model.predict_proba(X_test_flat)
    save_roc_pr_curves(y_test, y_probs_svm, "SVM", output_dir)
    plot_confusion_matrix(y_test, y_pred, "SVM", output_dir)

    joblib.dump(model, os.path.join(output_dir, 'svm_model.joblib'))
    print("💾 SVM model saved")
    
    monitor_memory_usage("after SVM")

def prepare_model_data(X_train, X_test, model_name, metadata):
    """
    Prepare data with correct shape for different model types
    Uses EEG structure information from metadata for proper reshaping
    """
    print(f"🔧 Preparing data for {model_name}...")
    print(f"   Original data shape: {X_train.shape}")
    
    if model_name == 'mlp':
        # MLP needs flattened input
        X_train_model = X_train.reshape(X_train.shape[0], -1)
        X_test_model = X_test.reshape(X_test.shape[0], -1)
        input_shape = (X_train_model.shape[1],)
        print(f"   MLP flattened shape: {X_train_model.shape}")
        
    else:
        # Use EEG structure for proper reshaping
        channels = metadata.get('eeg_channels', 32)
        timepoints = metadata.get('eeg_timepoints', 188)
        expected_features = channels * timepoints
        
        print(f"   EEG structure: {channels} channels × {timepoints} timepoints = {expected_features} features")
        
        # Verify data matches expected structure
        if X_train.shape[1] != expected_features:
            print(f"   ⚠️  Data shape mismatch! Expected {expected_features}, got {X_train.shape[1]}")
            print(f"   Using fallback reshaping...")
            # Fallback: treat as (samples, features, 1)
            X_train_model = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_model = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            input_shape = (X_train.shape[1], 1)
        else:
            # Proper EEG reshaping: (samples, timepoints, channels)
            X_train_model = X_train.reshape(-1, timepoints, channels)
            X_test_model = X_test.reshape(-1, timepoints, channels)
            input_shape = (timepoints, channels)
            print(f"   Reshaped to EEG structure: {X_train_model.shape}")
            print(f"   Input shape for model: {input_shape}")
    
    return X_train_model, X_test_model, input_shape

def create_enhanced_callbacks(model_name, output_dir, progress_position=1):
    """Create enhanced callbacks with smart early stopping and progress tracking"""
    
    callbacks = [
        # Enhanced progress tracking
        TrainingProgressCallback(model_name, position=progress_position),

        tf.keras.callbacks.TerminateOnNaN(),
        
        # Smart early stopping
        AdaptiveEarlyStopping(
            monitor='val_accuracy',
            min_delta=0.002,
            patience=8,
            baseline_patience=8,
            noise_tolerance=0.002,
            min_epochs=15,
            factor_patience=True,
            restore_best_weights=True
        ),
        
       ModelCheckpoint(
            filepath=os.path.join(output_dir, f"{model_name}_best.keras"),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=0
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks

def train_with_split(model_builder, X_train, y_train, X_test, y_test, output_dir, name, config, metadata):
    """Train model using simple train/validation split with enhanced features"""
    print(f"🚀 Training {name} with simple split validation...")
    
    # Memory monitoring
    monitor_memory_usage(f"before {name}")
    
    # Prepare data with correct shape for this model type
    X_train_model, X_test_model, input_shape = prepare_model_data(X_train, X_test, name, metadata)
    
    num_classes = len(np.unique(y_train))
    loss_fn = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

    # UPDATED: Use centralized class weight computation
    class_weight_dict = compute_class_weights(y_train)

    # Enhanced callbacks
    callbacks = create_enhanced_callbacks(name, output_dir, progress_position=1)

    # Prepare labels
    y_train_cat = y_train if num_classes == 2 else to_categorical(y_train, num_classes)
    
    # Build model
    model = model_builder(input_shape=input_shape, num_classes=num_classes, config=config)
    
    # UPDATED: Use centralized optimizer creation
    optimizer = create_optimizer(config.get('learning_rate', 1e-4))
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    print(f"📊 Model summary for {name}:")
    print(f"   Parameters: {model.count_params():,}")
    print(f"   Input shape: {input_shape}")

    # Train model
    start_time = time.time()
    history = model.fit(
        X_train_model, y_train_cat,
        validation_split=0.1,  # Use 10% of training data for validation
        epochs=RUNTIME_CONFIG['global']['epochs'],
        batch_size=config.get('batch_size', 64),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0  # Progress handled by our custom callback
    )
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.1f}s")

    # Save training curves and model
    save_training_curves(history, name, output_dir)
    model.save(os.path.join(output_dir, f'{name}_final.keras'))
    # validate the file exists
    if not os.path.exists(os.path.join(output_dir, f'{name}_final.keras')):
        print(f"⚠️  Model file not found: {os.path.join(output_dir, f'{name}_final.keras')}")

    # Evaluate on test set
    print("🔍 Evaluating on test set...")
    y_probs = model.predict(X_test_model, verbose=0)

    # ADD temperature scaling:
    print("🌡️  Applying temperature scaling...")
    y_probs_calibrated, temp_used = apply_dynamic_temperature_scaling(y_probs, y_test)
    y_pred = (y_probs_calibrated > 0.5).astype(int).flatten() if num_classes == 2 else np.argmax(y_probs, axis=1)

    print_evaluation(y_test, y_pred, name)
    plot_confusion_matrix(y_test, y_pred, "SVM", output_dir)

    # Update ROC/PR curves call:
    if num_classes == 2:
        save_roc_pr_curves(y_test, y_probs_calibrated, name, output_dir)
    else:
        save_roc_pr_curves(y_test, y_probs, name, output_dir)

    # Memory monitoring after training
    monitor_memory_usage(f"after {name}")
    
    print(f"✅ {name} model saved")
    return model

def analyze_group_balance_in_cv(X_train, y_train, groups_train, cv_splitter):
    """Analyze group balance across CV folds"""
    print(f"\n🔍 Analyzing group balance across CV folds...")
    
    fold_stats = []
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train, groups_train)):
        train_ecg = np.sum(y_train[train_idx] == 0)
        train_eeg = np.sum(y_train[train_idx] == 1) 
        val_ecg = np.sum(y_train[val_idx] == 0)
        val_eeg = np.sum(y_train[val_idx] == 1)
        
        train_groups = set([groups_train[i] for i in train_idx])
        val_groups = set([groups_train[i] for i in val_idx])
        
        print(f"   Fold {fold+1}: Train ECG={train_ecg}, EEG={train_eeg} ({train_ecg/(train_ecg+train_eeg)*100:.1f}% ECG)")
        print(f"           Val ECG={val_ecg}, EEG={val_eeg} ({val_ecg/(val_ecg+val_eeg)*100:.1f}% ECG)")
        print(f"           Groups: {len(train_groups)} train, {len(val_groups)} val")
        
        fold_stats.append({
            'train_ecg_pct': train_ecg/(train_ecg+train_eeg)*100,
            'val_ecg_pct': val_ecg/(val_ecg+val_eeg)*100
        })
    
    # Check balance consistency
    train_ecg_pcts = [f['train_ecg_pct'] for f in fold_stats]
    val_ecg_pcts = [f['val_ecg_pct'] for f in fold_stats]
    
    print(f"   📊 ECG% variance across folds:")
    print(f"      Train: {np.std(train_ecg_pcts):.1f}% std (should be <5%)")
    print(f"      Val: {np.std(val_ecg_pcts):.1f}% std (should be <5%)")
    
    if np.std(train_ecg_pcts) > 5 or np.std(val_ecg_pcts) > 5:
        print(f"   ⚠️  HIGH variance in class balance across folds!")
        print(f"   💡 Consider stratified GroupKFold or more groups")
    
    return fold_stats


def train_with_kfold(model_builder, X_train, y_train, X_test, y_test, output_dir, name, config, metadata, groups_train=None):
    """Train model using K-fold cross-validation with enhanced features and group-based CV"""
    print(f"🚀 Training {name} with {RUNTIME_CONFIG['global']['n_splits']}-fold cross-validation...")
    
    # Memory monitoring
    monitor_memory_usage(f"before {name} k-fold")
    
    # Prepare data with correct shape for this model type
    X_train_model, X_test_model, input_shape = prepare_model_data(X_train, X_test, name, metadata)
    
    num_classes = len(np.unique(y_train))
    loss_fn = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

    # UPDATED: Use centralized class weight computation
    class_weight_dict = compute_class_weights(y_train)

    # FIXED: Group-based cross-validation to prevent subject leakage
    if groups_train is not None:
        print(f"🔒 Using StratifiedGroupKFold for better class balance")
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            cv_splitter = StratifiedGroupKFold(
                n_splits=RUNTIME_CONFIG['global']['n_splits'], 
                shuffle=True, 
                random_state=RUNTIME_CONFIG['global']['random_state']
            )
            print(f"   ✅ Using StratifiedGroupKFold with {RUNTIME_CONFIG['global']['n_splits']} folds")
        except ImportError:
            print(f"   ⚠️  StratifiedGroupKFold not available (needs scikit-learn ≥1.4)")
            print(f"   📦 Upgrade with: pip install scikit-learn>=1.4")
            # Fallback to GroupKFold
            from sklearn.model_selection import GroupKFold
            cv_splitter = GroupKFold(n_splits=RUNTIME_CONFIG['global']['n_splits'])
        cv_iterator = cv_splitter.split(X_train_model, y_train, groups_train)
        
        # Validate groups
        unique_groups = len(set(groups_train))
        print(f"   📊 Found {unique_groups} unique groups for CV")
        if unique_groups < RUNTIME_CONFIG['global']['n_splits']:
            print(f"   ⚠️  Warning: Only {unique_groups} groups for {RUNTIME_CONFIG['global']['n_splits']} folds!")
        
        # NEW: Analyze group balance across folds
        analyze_group_balance_in_cv(X_train_model, y_train, groups_train, cv_splitter)
        
        # Reset iterator after analysis
        cv_iterator = cv_splitter.split(X_train_model, y_train, groups_train)
        
    else:
        print(f"⚠️  No groups provided - using StratifiedKFold (potential leakage)")
        from sklearn.model_selection import StratifiedKFold
        cv_splitter = StratifiedKFold(n_splits=RUNTIME_CONFIG['global']['n_splits'], shuffle=True, random_state=RUNTIME_CONFIG['global']['random_state'])
        cv_iterator = cv_splitter.split(X_train_model, y_train)

    fold_scores = []

    # Progress bar for folds
    fold_progress = tqdm(
        total=RUNTIME_CONFIG['global']['n_splits'], 
        desc=f"🔄 {name} K-Fold", 
        position=0,
        leave=True
    )

    for fold, (train_idx, val_idx) in enumerate(cv_iterator):
        fold_progress.set_description(f"🔄 {name} Fold {fold + 1}")
        
        X_tr, X_val = X_train_model[train_idx], X_train_model[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # ADDED: Group validation for debugging
        if groups_train is not None:
            train_groups = set([groups_train[i] for i in train_idx])
            val_groups = set([groups_train[i] for i in val_idx])
            overlap = train_groups.intersection(val_groups)
            if overlap:
                print(f"   ⚠️  WARNING: Group overlap detected in fold {fold + 1}: {overlap}")
            else:
                print(f"   ✅ Fold {fold + 1}: No group overlap ({len(train_groups)} train, {len(val_groups)} val groups)")

        if num_classes > 2:
            y_tr = to_categorical(y_tr, num_classes)
            y_val = to_categorical(y_val, num_classes)

        model = model_builder(input_shape=input_shape, num_classes=num_classes, config=config)
        
        # UPDATED: Use centralized optimizer creation
        optimizer = create_optimizer(config.get('learning_rate', 1e-4))
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        
        # Enhanced callbacks for this fold
        callbacks = create_enhanced_callbacks(f"{name}_fold{fold+1}", output_dir, progress_position=1)

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=RUNTIME_CONFIG['global']['epochs'],
            batch_size=config.get('batch_size', 64),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate on validation set
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        model.save(os.path.join(output_dir, f'{name}_fold{fold+1}.keras'))

        save_training_curves(history, f'{name}_fold{fold+1}', output_dir)

        fold_scores.append(val_acc)
        fold_progress.set_postfix({
            'Fold_Acc': f"{val_acc:.4f}",
            'Mean_CV': f"{np.mean(fold_scores):.4f}" if fold_scores else "N/A"
        })
        fold_progress.update(1)
        
        # Cleanup between folds
        del model, X_tr, X_val, y_tr, y_val
        gc.collect()

    fold_progress.close()

    print(f"\n✅ K-Fold Cross-Validation Results for {name}:")
    print(f"   Mean CV Score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"   Individual Fold Scores: {[f'{score:.4f}' for score in fold_scores]}")
    
    # NEW: Check for high variance in CV scores
    cv_std = np.std(fold_scores)
    cv_mean = np.mean(fold_scores)
    if cv_std > 0.1:  # High variance threshold
        print(f"   ⚠️  HIGH CV variance detected: {cv_std:.4f}")
        print(f"   💡 Possible causes: overfitting, group imbalance, or insufficient regularization")
        if name == 'mlp':
            print(f"   💡 For MLP: Consider increasing L2 regularization or dropout")
    
    log_fold_metrics_to_csv(name, fold_scores, output_dir)

    # Train final model on full training set
    print(f"\nTraining final {name} model on full training set...")
    y_train_final = y_train if num_classes == 2 else to_categorical(y_train, num_classes)
    final_model = model_builder(input_shape=input_shape, num_classes=num_classes, config=config)
    
    # UPDATED: Use centralized optimizer creation
    optimizer = create_optimizer(config.get('learning_rate', 1e-4))
    final_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Enhanced callbacks for final training
    callbacks = create_enhanced_callbacks(f"{name}_final", output_dir, progress_position=0)
    
    final_history = final_model.fit(
        X_train_model, y_train_final,
        validation_split=0.1,
        epochs=RUNTIME_CONFIG['global']['epochs'],
        batch_size=config.get('batch_size', 64),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=0
    )
    
    # Save final model and evaluate
    final_model.save(os.path.join(output_dir, f'{name}_final.keras'))
    save_training_curves(final_history, name, output_dir)
    
    # Evaluate on test set
    print("🔍 Evaluating final model on test set...")
    y_probs = final_model.predict(X_test_model, verbose=0)

    # ADD TEMPERATURE SCALING HERE:
    print("🌡️  Applying temperature scaling for calibration...")
    y_probs_calibrated, temp_used = apply_dynamic_temperature_scaling(y_probs, y_test)

    if num_classes == 2:
        y_pred = (y_probs_calibrated > 0.5).astype(int)
        # Use calibrated probs for ROC/PR curves
        save_roc_pr_curves(y_test, y_probs_calibrated, name, output_dir)
    else:
        y_pred = np.argmax(y_probs, axis=1)
        save_roc_pr_curves(y_test, y_probs, name, output_dir)

    print_evaluation(y_test, y_pred, name)
    plot_confusion_matrix(y_test, y_pred, name, output_dir)
    
    # Memory monitoring after training
    monitor_memory_usage(f"after {name} k-fold")
    
    print(f"✅ {name} model saved")
    return final_model

def train_keras_model(model_builder, X_train, y_train, X_test, y_test, output_dir, name, config, metadata, validation_strategy='auto', groups_train=None):
    """
    Unified training function for Keras models with automatic strategy selection
    """
    # Choose validation strategy
    actual_strategy = choose_validation_strategy(
        dataset_size=len(X_train),
        time_budget='medium',
        force_strategy=validation_strategy if validation_strategy != 'auto' else None
    )
    
    print(f"📊 Dataset size: {len(X_train):,} samples")
    print(f"🎯 Using validation strategy: {actual_strategy}")

    log_model_configuration(name, config)
    
    if actual_strategy == 'kfold':
        return train_with_kfold(model_builder, X_train, y_train, X_test, y_test, output_dir, name, config, metadata, groups_train)
    else:
        return train_with_split(model_builder, X_train, y_train, X_test, y_test, output_dir, name, config, metadata)

def train_dual_branch(X_train, y_train, X_test, y_test, output_dir, metadata, validation_strategy='auto', groups_train=None):
    """
    Train Dual-Branch CNN with proper CV balance and enhanced features
    FIXED: Uses StratifiedGroupKFold for balanced class distribution
    FIXED: Saves per-fold training curves
    ENHANCED: Includes group balance analysis
    """
    print("\n🚀 Training Dual-Branch CNN...")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Memory monitoring
    monitor_memory_usage("before dual_branch")

    # Prepare data for dual branch (needs 3D input)
    X_train_model, X_test_model, input_shape = prepare_model_data(X_train, X_test, 'dual_branch', metadata)
    
    # Choose validation strategy
    actual_strategy = choose_validation_strategy(
        dataset_size=len(X_train),
        time_budget='medium',
        force_strategy=validation_strategy if validation_strategy != 'auto' else None
    )
    
    print(f"📊 Dataset size: {len(X_train):,} samples")
    print(f"🎯 Using validation strategy: {actual_strategy}")

    log_model_configuration('dual_branch', RUNTIME_CONFIG['dual_branch'])

    if actual_strategy == 'kfold':
        # FIXED: Enhanced Group-based cross-validation with proper stratification
        if groups_train is not None:
            print(f"🔒 Using Enhanced GroupKFold with class balance awareness")
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                cv_splitter = StratifiedGroupKFold(
                    n_splits=RUNTIME_CONFIG['global']['n_splits'], 
                    shuffle=True, 
                    random_state=RUNTIME_CONFIG['global']['random_state']
                )
                print(f"   ✅ Using StratifiedGroupKFold for balanced class distribution")
            except ImportError:
                print(f"   ⚠️  StratifiedGroupKFold not available (needs scikit-learn ≥1.4)")
                print(f"   📦 Falling back to GroupKFold")
                from sklearn.model_selection import GroupKFold
                cv_splitter = GroupKFold(n_splits=RUNTIME_CONFIG['global']['n_splits'])
            
            cv_iterator = cv_splitter.split(X_train_model, y_train, groups_train)
            
            # Validate groups
            unique_groups = len(set(groups_train))
            print(f"   📊 Found {unique_groups} unique groups for CV")
            if unique_groups < RUNTIME_CONFIG['global']['n_splits']:
                print(f"   ⚠️  Warning: Only {unique_groups} groups for {RUNTIME_CONFIG['global']['n_splits']} folds!")
            
            # ADDED: Analyze group balance across folds (like other models)
            analyze_group_balance_in_cv(X_train_model, y_train, groups_train, cv_splitter)
            
            # Reset iterator after analysis
            cv_iterator = cv_splitter.split(X_train_model, y_train, groups_train)
            
        else:
            print(f"⚠️  No groups provided - using StratifiedKFold (potential leakage)")
            from sklearn.model_selection import StratifiedKFold
            cv_splitter = StratifiedKFold(n_splits=RUNTIME_CONFIG['global']['n_splits'], shuffle=True, random_state=RUNTIME_CONFIG['global']['random_state'])
            cv_iterator = cv_splitter.split(X_train_model, y_train)

        fold_scores = []
        
        # Progress bar for folds
        fold_progress = tqdm(
            total=RUNTIME_CONFIG['global']['n_splits'], 
            desc="🔄 Dual-Branch K-Fold", 
            position=0,
            leave=True
        )
        
        for fold, (train_idx, val_idx) in enumerate(cv_iterator):
            fold_progress.set_description(f"🔄 Dual-Branch Fold {fold + 1}")
            
            X_tr, X_val = X_train_model[train_idx], X_train_model[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # ENHANCED: Group validation for debugging with detailed logging
            if groups_train is not None:
                train_groups = set([groups_train[i] for i in train_idx])
                val_groups = set([groups_train[i] for i in val_idx])
                overlap = train_groups.intersection(val_groups)
                if overlap:
                    print(f"   ⚠️  WARNING: Group overlap detected in fold {fold + 1}: {overlap}")
                else:
                    print(f"   ✅ Fold {fold + 1}: No group overlap ({len(train_groups)} train, {len(val_groups)} val groups)")
                
                # ADDED: Class balance analysis per fold
                train_ecg = np.sum(y_tr == 0)
                train_eeg = np.sum(y_tr == 1)
                val_ecg = np.sum(y_val == 0)
                val_eeg = np.sum(y_val == 1)
                
                train_ecg_pct = train_ecg / (train_ecg + train_eeg) * 100
                val_ecg_pct = val_ecg / (val_ecg + val_eeg) * 100
                
                print(f"       Train: {train_ecg:,} ECG, {train_eeg:,} EEG ({train_ecg_pct:.1f}% ECG)")
                print(f"       Val:   {val_ecg:,} ECG, {val_eeg:,} EEG ({val_ecg_pct:.1f}% ECG)")

            # Build model for this fold
            model = build_dual_branch_cnn(input_shape=input_shape, num_classes=2, config=RUNTIME_CONFIG['dual_branch'])
            
            # UPDATED: Use centralized optimizer creation
            optimizer = create_optimizer(RUNTIME_CONFIG['dual_branch']['learning_rate'])
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Enhanced callbacks for this fold with proper position
            callbacks = create_enhanced_callbacks(f"dual_branch_fold{fold+1}", output_dir, progress_position=1)

            # UPDATED: Use centralized class weight computation
            class_weight_dict = compute_class_weights(y_tr)

            # Train model for this fold
            start_time = time.time()
            history = model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=RUNTIME_CONFIG['global']['epochs'],
                batch_size=RUNTIME_CONFIG['dual_branch']['batch_size'],
                class_weight=class_weight_dict,  # ADDED: Class balancing
                callbacks=callbacks,
                verbose=0
            )
            fold_time = time.time() - start_time
            
            # FIXED: Save per-fold training curves (was missing!)
            save_training_curves(history, f'dual_branch_fold{fold+1}', output_dir)
            
            # Evaluate on validation set
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            
            # ADDED: Save fold model
            model.save(os.path.join(output_dir, f'dual_branch_fold{fold+1}.keras'))
            
            fold_scores.append(val_acc)
            fold_progress.set_postfix({
                'Fold_Acc': f"{val_acc:.4f}",
                'Mean_CV': f"{np.mean(fold_scores):.4f}" if fold_scores else "N/A",
                'Time': f"{fold_time:.1f}s"
            })
            fold_progress.update(1)
            
            # Cleanup between folds
            del model, X_tr, X_val, y_tr, y_val
            gc.collect()

        fold_progress.close()

        print(f"\n✅ Dual-Branch CNN K-Fold Results:")
        print(f"   Mean CV Score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        print(f"   Individual Fold Scores: {[f'{score:.4f}' for score in fold_scores]}")
        
        # ADDED: Check for high variance in CV scores (like other models)
        cv_std = np.std(fold_scores)
        cv_mean = np.mean(fold_scores)
        if cv_std > 0.1:  # High variance threshold
            print(f"   ⚠️  HIGH CV variance detected: {cv_std:.4f}")
            print(f"   💡 Possible causes: overfitting, group imbalance, or insufficient regularization")
            print(f"   💡 For Dual-Branch: Consider increasing L2 regularization or dropout")
        
        # ADDED: Save fold metrics to CSV
        log_fold_metrics_to_csv('dual_branch', fold_scores, output_dir)

        # Train final model on full training set
        print(f"\nTraining final dual-branch model on full training set...")
        final_model = build_dual_branch_cnn(input_shape=input_shape, num_classes=2, config=RUNTIME_CONFIG['dual_branch'])
        
        # UPDATED: Use centralized optimizer creation
        optimizer = create_optimizer(RUNTIME_CONFIG['dual_branch']['learning_rate'])
        final_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Enhanced callbacks for final training
        callbacks = create_enhanced_callbacks("dual_branch_final", output_dir, progress_position=0)

        # UPDATED: Use centralized class weight computation
        class_weight_dict = compute_class_weights(y_train)

        final_start_time = time.time()
        final_history = final_model.fit(
            X_train_model, y_train,
            validation_split=0.1,
            epochs=RUNTIME_CONFIG['global']['epochs'],
            batch_size=RUNTIME_CONFIG['dual_branch']['batch_size'],
            class_weight=class_weight_dict,  # ADDED: Class balancing
            callbacks=callbacks,
            verbose=0
        )
        final_time = time.time() - final_start_time
        print(f"⏱️  Final training completed in {final_time:.1f}s")

        # Save final training curves
        save_training_curves(final_history, 'dual_branch', output_dir)
        
        # Evaluate on test set
        print("🔍 Evaluating final model on test set...")
        y_probs = final_model.predict(X_test_model, verbose=0)
        
        # ENHANCED: Apply temperature scaling for calibration
        print("🌡️  Applying temperature scaling for dual-branch...")
        y_probs_calibrated, temp_used = apply_dynamic_temperature_scaling(y_probs, y_test)
        y_pred = (y_probs_calibrated > 0.5).astype(int).flatten()
        
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["ECG", "EEG"]))
        plot_confusion_matrix(y_test, y_pred, "dual_branch", output_dir)
        
        # Use calibrated probabilities for ROC/PR curves
        save_roc_pr_curves(y_test, y_probs_calibrated, 'dual_branch', output_dir)

        # Save final model
        final_model.save(os.path.join(output_dir, 'dual_branch_final.keras'))

    else:  # Simple split validation
        print("🎯 Using simple split validation...")
        model = build_dual_branch_cnn(input_shape=input_shape, num_classes=2, config=RUNTIME_CONFIG['dual_branch'])
        
        # UPDATED: Use centralized optimizer creation
        optimizer = create_optimizer(RUNTIME_CONFIG['dual_branch']['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        print(f"📊 Model summary for dual_branch:")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Input shape: {input_shape}")

        # Enhanced callbacks
        callbacks = create_enhanced_callbacks("dual_branch", output_dir, progress_position=0)

        # UPDATED: Use centralized class weight computation
        class_weight_dict = compute_class_weights(y_train)

        start_time = time.time()
        history = model.fit(
            X_train_model, y_train,
            validation_split=0.1,
            epochs=RUNTIME_CONFIG['global']['epochs'],
            batch_size=RUNTIME_CONFIG['dual_branch']['batch_size'],
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=0
        )
        
        training_time = time.time() - start_time
        print(f"⏱️  Training completed in {training_time:.1f}s")

        # Save training curves and model
        save_training_curves(history, 'dual_branch', output_dir)
        
        # Evaluate on test set
        print("🔍 Evaluating on test set...")
        y_probs = model.predict(X_test_model, verbose=0)
        
        # ENHANCED: Apply temperature scaling for calibration
        print("🌡️  Applying temperature scaling for dual-branch...")
        y_probs_calibrated, temp_used = apply_dynamic_temperature_scaling(y_probs, y_test)
        y_pred = (y_probs_calibrated > 0.5).astype(int).flatten()
        
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["ECG", "EEG"]))
        plot_confusion_matrix(y_test, y_pred, "dual_branch", output_dir)
        
        # Use calibrated probabilities for ROC/PR curves
        save_roc_pr_curves(y_test, y_probs_calibrated, 'dual_branch', output_dir)
        
        # Save model
        model.save(os.path.join(output_dir, 'dual_branch_final.keras'))

    # Memory monitoring after training
    monitor_memory_usage("after dual_branch")
    print("✅ Dual-Branch CNN training complete.")
    
    return final_model if actual_strategy == 'kfold' else model

def find_optimal_temperature(y_true, y_probs, validation_split=0.1):
    """
    Find optimal temperature using validation data to minimize calibration error
    
    Args:
        y_true: True labels
        y_probs: Raw model probabilities
        validation_split: Fraction to use for temperature optimization
        
    Returns:
        optimal_temperature: Best temperature found
        calibration_info: Dict with calibration metrics
    """
    from sklearn.model_selection import train_test_split
    from scipy.optimize import minimize_scalar
    import numpy as np
    
    # Split data for temperature optimization
    if len(y_true) > 100:  # Only if we have enough data
        _, y_val, _, probs_val = train_test_split(
            y_true, y_probs, test_size=validation_split, 
            stratify=y_true, random_state=42
        )
    else:
        # Use all data if dataset is small
        y_val, probs_val = y_true, y_probs
    
    def calibration_loss(temperature):
        """Calculate Expected Calibration Error (ECE) for given temperature"""
        # Apply temperature scaling
        epsilon = 1e-8
        if len(probs_val.shape) > 1 and probs_val.shape[1] > 1:
            probs_pos = probs_val[:, 1]
        else:
            probs_pos = probs_val.flatten()
        
        # Convert to logits and apply temperature
        probs_clipped = np.clip(probs_pos, epsilon, 1 - epsilon)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        calibrated_logits = logits / temperature
        calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
        
        # Calculate Expected Calibration Error (ECE)
        ece = calculate_ece(y_val, calibrated_probs)
        return ece
    
    # Search for optimal temperature
    try:
        result = minimize_scalar(
            calibration_loss, 
            bounds=(0.1, 10.0),  # Reasonable temperature range
            method='bounded'
        )
        optimal_temp = result.x
        best_ece = result.fun
        
        # Validate the result
        if optimal_temp < 0.1 or optimal_temp > 10.0:
            print(f"   ⚠️  Optimal temperature {optimal_temp:.2f} outside bounds, using fallback")
            optimal_temp = 1.5  # Conservative fallback
            
    except Exception as e:
        print(f"   ⚠️  Temperature optimization failed: {e}")
        optimal_temp = 1.5  # Conservative fallback
        best_ece = None
    
    # Calculate calibration metrics for reporting
    calibration_info = {
        'optimal_temperature': optimal_temp,
        'ece_before': calibration_loss(1.0),  # ECE without scaling
        'ece_after': calibration_loss(optimal_temp),
        'validation_samples': len(y_val)
    }
    
    return optimal_temp, calibration_info

def calculate_ece(y_true, y_probs, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Points in this bin
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_probs[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def apply_dynamic_temperature_scaling(y_probs, y_true=None, temperature=None):
    """
    Apply temperature scaling with dynamic temperature optimization
    
    Args:
        y_probs: Raw model probabilities
        y_true: True labels (for temperature optimization)
        temperature: Fixed temperature (if None, will optimize)
        
    Returns:
        calibrated_probs: Temperature-scaled probabilities
        temperature_used: The temperature that was applied
    """
    import numpy as np
    
    # If temperature not provided, try to find optimal
    if temperature is None and y_true is not None:
        print("🌡️  Finding optimal temperature...")
        optimal_temp, cal_info = find_optimal_temperature(y_true, y_probs)
        
        print(f"   🎯 Optimal temperature: {optimal_temp:.2f}")
        print(f"   📊 ECE before: {cal_info['ece_before']:.4f}")
        print(f"   📊 ECE after: {cal_info['ece_after']:.4f}")
        print(f"   📊 Validation samples: {cal_info['validation_samples']}")
        
        temperature = optimal_temp
    elif temperature is None:
        # Fallback when no ground truth available
        print("🌡️  No ground truth available, using conservative temperature")
        temperature = 1.5
    
    # Apply temperature scaling
    epsilon = 1e-8
    if len(y_probs.shape) > 1 and y_probs.shape[1] > 1:
        y_probs_pos = y_probs[:, 1]
    else:
        y_probs_pos = y_probs.flatten()
    
    # Convert to logits and apply temperature
    y_probs_clipped = np.clip(y_probs_pos, epsilon, 1 - epsilon)
    logits = np.log(y_probs_clipped / (1 - y_probs_clipped))
    calibrated_logits = logits / temperature
    calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
    
    print(f"   🌡️  Temperature applied: {temperature:.2f}")
    print(f"   📊 Before: range=[{y_probs_pos.min():.4f}, {y_probs_pos.max():.4f}]")
    print(f"   📊 After:  range=[{calibrated_probs.min():.4f}, {calibrated_probs.max():.4f}]")
    
    return calibrated_probs, temperature

# ============================================================================
# ENHANCED MAIN TRAINING PIPELINE WITH MEMORY MANAGEMENT
# ============================================================================

def train_all_models_with_management(models_to_train, X_train, y_train, X_test, y_test, output_dir, metadata, validation_strategy='auto'):
    """
    Train all models with proper memory management and progress tracking
    """
    print(f"\n🚀 Training {len(models_to_train)} models with enhanced pipeline...")
    print("=" * 70)
    
    # Overall progress tracking
    overall_start_time = time.time()
    results = {}
    
    # Overall progress bar
    overall_progress = tqdm(
        total=len(models_to_train), 
        desc="🎯 Training Pipeline", 
        position=0,
        leave=True,
        unit="model"
    )
    
    builders = {
        'svm': train_svm,
        'simple_cnn': build_simple_cnn,
        'cnn_lstm': build_cnn_lstm,
        'mlp': build_mlp,
        'tcn': build_tcn,
        'dual_branch': train_dual_branch
    }
    
    for i, model_name in enumerate(models_to_train):
        model_start_time = time.time()
        
        # Update overall progress
        overall_progress.set_description(f"🎯 Training {model_name}")
        cpu_mem, gpu_mem = get_memory_usage()
        overall_progress.set_postfix({
            'Model': model_name,
            'Progress': f"{i+1}/{len(models_to_train)}",
            'CPU': f"{cpu_mem:.1f}GB",
            'GPU': f"{gpu_mem:.1f}GB" if gpu_mem > 0 else "N/A"
        })
        
        print(f"\n{'='*20} {model_name.upper()} ({'='*20}")
        
        try:
            # Train specific model
            if model_name == 'svm':
                train_svm(X_train, y_train, X_test, y_test, output_dir)
            elif model_name == 'dual_branch':
                train_dual_branch(X_train, y_train, X_test, y_test, output_dir, metadata, validation_strategy)
            else:
                train_keras_model(builders[model_name], X_train, y_train, X_test, y_test, 
                                output_dir, model_name, RUNTIME_CONFIG[model_name], 
                                metadata, validation_strategy)
            
            # Record results
            model_time = time.time() - model_start_time
            results[model_name] = {
                'training_time': model_time,
                'status': 'success'
            }
            
            print(f"✅ {model_name} completed in {model_time:.1f}s")

            # =====================================
            print(f"🔍 Running feature importance analysis for {model_name}...")
            try:
                if model_name == 'svm':
                    # Load the saved SVM model
                    import joblib
                    svm_model_path = os.path.join(output_dir, 'svm_model.joblib')
                    if os.path.exists(svm_model_path):
                        svm_model = joblib.load(svm_model_path)
                        importance_scores = analyze_feature_importance(svm_model, X_test, y_test, model_name)
                    else:
                        print(f"   ⚠️  SVM model file not found, skipping importance analysis")
                        importance_scores = []
                else:
                    # Load the saved Keras model
                    from tensorflow.keras.models import load_model
                    keras_model_path = os.path.join(output_dir, f'{model_name}_final.keras')
                    if os.path.exists(keras_model_path):
                        keras_model = load_model(keras_model_path)
                        importance_scores = analyze_feature_importance(keras_model, X_test, y_test, model_name)
                        # Clean up model from memory
                        del keras_model
                    else:
                        print(f"   ⚠️  Keras model file not found, skipping importance analysis")
                        importance_scores = []
                
                # Save importance scores
                if importance_scores:
                    import pickle
                    importance_path = os.path.join(output_dir, f'{model_name}_feature_importance.pkl')
                    with open(importance_path, 'wb') as f:
                        pickle.dump(importance_scores, f)
                    print(f"   💾 Feature importance saved to: {importance_path}")
                
            except Exception as importance_error:
                print(f"   ⚠️  Feature importance analysis failed: {importance_error}")
            # =====================================
            
        except Exception as e:
            model_time = time.time() - model_start_time
            results[model_name] = {
                'training_time': model_time,
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ {model_name} failed after {model_time:.1f}s: {str(e)}")
        
        # Cleanup between models (crucial for memory management)
        print("🧹 Cleaning up before next model...")
        cleanup_between_models()
        
        # Update progress
        overall_progress.update(1)
        elapsed_total = time.time() - overall_start_time
        remaining_estimate = estimate_remaining_time(i, len(models_to_train), overall_start_time)
        overall_progress.set_postfix({
            'Completed': f"{i+1}/{len(models_to_train)}",
            'Elapsed': f"{elapsed_total/60:.1f}m",
            'Remaining': remaining_estimate,
            'Last': f"{model_time:.1f}s"
        })
    
    overall_progress.close()
    
    # Final summary
    total_time = time.time() - overall_start_time
    print(f"\n🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Results summary:")
    
    successful_models = [m for m, r in results.items() if r['status'] == 'success']
    failed_models = [m for m, r in results.items() if r['status'] == 'failed']
    
    print(f"   ✅ Successful: {len(successful_models)}/{len(models_to_train)}")
    print(f"   ❌ Failed: {len(failed_models)}/{len(models_to_train)}")
    
    if successful_models:
        print(f"   🚀 Fastest: {min(results.items(), key=lambda x: x[1]['training_time'])[0]}")
        print(f"   🐌 Slowest: {max(results.items(), key=lambda x: x[1]['training_time'])[0]}")
    
    if failed_models:
        print(f"   💥 Failed models: {', '.join(failed_models)}")
    
    # Final memory status
    monitor_memory_usage("final")
    
    return results

def analyze_feature_importance(model, X_test, y_test, model_name, top_k=20):
    """
    ENHANCED: Feature importance with detailed debugging
    """
    print(f"🔍 Analyzing feature importance for {model_name}...")
    
    # Prepare data according to model type
    if model_name == 'svm':
        model_obj = model
        # X_for_analysis = scaler.transform(X_test)
        
        def predict_fn(X):
            return model_obj.predict_proba(X)[:, 1]
            
    elif model_name == 'mlp':
        X_for_analysis = X_test
        
        def predict_fn(X):
            return model.predict(X, verbose=0).flatten()
            
    elif model_name in ['simple_cnn', 'cnn_lstm', 'tcn', 'dual_branch']:
        if X_test.shape[1] == 6016:
            X_for_analysis = X_test.reshape(X_test.shape[0], 188, 32)
        else:
            X_for_analysis = X_test
        
        def predict_fn(X):
            return model.predict(X, verbose=0).flatten()
    
    print(f"   📊 Input shape: {X_for_analysis.shape}")
    
    # DEBUGGING: Check model performance first
    try:
        baseline_predictions = predict_fn(X_for_analysis)
        baseline_acc = np.mean((baseline_predictions > 0.5) == y_test)
        
        # Check prediction distribution
        pred_mean = np.mean(baseline_predictions)
        pred_std = np.std(baseline_predictions)
        pred_min = np.min(baseline_predictions)
        pred_max = np.max(baseline_predictions)
        
        # Check how many predictions are near extremes
        near_zero = np.sum(baseline_predictions < 0.1)
        near_one = np.sum(baseline_predictions > 0.9)
        middle_range = len(baseline_predictions) - near_zero - near_one
        
        print(f"   📊 BASELINE ANALYSIS:")
        print(f"      Accuracy: {baseline_acc:.4f}")
        print(f"      Pred mean: {pred_mean:.4f}, std: {pred_std:.4f}")
        print(f"      Pred range: [{pred_min:.4f}, {pred_max:.4f}]")
        print(f"      Near 0 (<0.1): {near_zero:,} ({near_zero/len(baseline_predictions)*100:.1f}%)")
        print(f"      Near 1 (>0.9): {near_one:,} ({near_one/len(baseline_predictions)*100:.1f}%)")
        print(f"      Middle range: {middle_range:,} ({middle_range/len(baseline_predictions)*100:.1f}%)")
        
        # If predictions are too extreme, feature importance won't work
        if (near_zero + near_one) / len(baseline_predictions) > 0.9:
            print(f"   ⚠️  WARNING: {(near_zero + near_one)/len(baseline_predictions)*100:.1f}% predictions are extreme!")
            print(f"   💡 Model is overconfident - feature importance will be minimal")
            print(f"   💡 This indicates the task is too easy or model is overfitting")
            
            if baseline_acc > 0.95:
                print(f"   🚨 CRITICAL: Model has {baseline_acc*100:.1f}% accuracy - task is too easy!")
                return []
        
    except Exception as e:
        print(f"   ❌ Baseline prediction failed: {e}")
        return []
    
    # Only proceed if baseline is reasonable
    if baseline_acc < 0.6 or baseline_acc > 0.98:
        print(f"   ⚠️  Baseline accuracy {baseline_acc:.4f} is too extreme for meaningful feature importance")
        return []
    
    # Test on smaller subset for speed
    n_samples = min(200, len(X_test))  # Even smaller subset
    test_indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_subset = X_for_analysis[test_indices]
    y_subset = y_test[test_indices]
    
    # Get baseline for subset
    baseline_subset_pred = predict_fn(X_subset)
    baseline_subset_acc = np.mean((baseline_subset_pred > 0.5) == y_subset)
    
    print(f"   📊 Testing on {n_samples} samples, subset baseline: {baseline_subset_acc:.4f}")
    
    feature_importance = []
    
    # Test fewer features with more detailed logging
    if model_name in ['simple_cnn', 'cnn_lstm', 'tcn', 'dual_branch']:
        print(f"   🔄 Testing channel importance (32 channels)...")
        
        # Test all 32 channels
        for channel_idx in range(32):
            X_permuted = X_subset.copy()
            
            # Permute this entire channel
            for sample_idx in range(X_subset.shape[0]):
                original_channel = X_permuted[sample_idx, :, channel_idx].copy()
                X_permuted[sample_idx, :, channel_idx] = np.random.permutation(original_channel)
            
            try:
                permuted_pred = predict_fn(X_permuted)
                permuted_acc = np.mean((permuted_pred > 0.5) == y_subset)
                importance = baseline_subset_acc - permuted_acc
                
                # DEBUGGING: Log first few channels in detail
                if channel_idx < 5:
                    pred_change = np.mean(np.abs(permuted_pred - baseline_subset_pred))
                    print(f"      Ch{channel_idx:02d}: baseline_acc={baseline_subset_acc:.4f}, "
                          f"permuted_acc={permuted_acc:.4f}, importance={importance:.4f}, "
                          f"avg_pred_change={pred_change:.4f}")
                
                feature_importance.append((f"Channel_{channel_idx:02d}", importance))
                
            except Exception as e:
                print(f"      ❌ Channel {channel_idx} failed: {e}")
                feature_importance.append((f"Channel_{channel_idx:02d}", 0.0))
        
        # Test some timesteps too
        print(f"   🔄 Testing timestep importance (sample of 10 timesteps)...")
        timestep_samples = np.random.choice(188, 10, replace=False)
        
        for timestep_idx in timestep_samples:
            X_permuted = X_subset.copy()
            
            # Permute this timestep across all channels
            for sample_idx in range(X_subset.shape[0]):
                original_timestep = X_permuted[sample_idx, timestep_idx, :].copy()
                X_permuted[sample_idx, timestep_idx, :] = np.random.permutation(original_timestep)
            
            try:
                permuted_pred = predict_fn(X_permuted)
                permuted_acc = np.mean((permuted_pred > 0.5) == y_subset)
                importance = baseline_subset_acc - permuted_acc
                feature_importance.append((f"Timestep_{timestep_idx:03d}", importance))
                
            except Exception as e:
                feature_importance.append((f"Timestep_{timestep_idx:03d}", 0.0))
    
    else:  # SVM, MLP
        print(f"   🔄 Testing individual features (sample of 50)...")
        feature_indices = np.random.choice(X_subset.shape[1], 50, replace=False)
        
        for i, feature_idx in enumerate(feature_indices):
            X_permuted = X_subset.copy()
            original_values = X_permuted[:, feature_idx].copy()
            X_permuted[:, feature_idx] = np.random.permutation(original_values)
            
            try:
                permuted_pred = predict_fn(X_permuted)
                permuted_acc = np.mean((permuted_pred > 0.5) == y_subset)
                importance = baseline_subset_acc - permuted_acc
                
                # Decode feature
                channel = feature_idx // 188
                timepoint = feature_idx % 188
                feature_name = f"Ch{channel:02d}_T{timepoint:03d}"
                
                # DEBUGGING: Log first few features
                if i < 5:
                    pred_change = np.mean(np.abs(permuted_pred - baseline_subset_pred))
                    print(f"      {feature_name}: baseline={baseline_subset_acc:.4f}, "
                          f"permuted={permuted_acc:.4f}, importance={importance:.4f}, "
                          f"pred_change={pred_change:.4f}")
                
                feature_importance.append((feature_name, importance))
                
            except Exception as e:
                feature_importance.append((f"Feature_{feature_idx}", 0.0))
    
    # Sort and display results
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    meaningful_features = [f for f in feature_importance if f[1] > 0.001]
    
    print(f"\n📊 FEATURE IMPORTANCE RESULTS for {model_name}:")
    print(f"   Total features tested: {len(feature_importance)}")
    print(f"   Meaningful features (>0.001): {len(meaningful_features)}")
    
    if len(meaningful_features) == 0:
        print(f"   ⚠️  NO meaningful features found!")
        print(f"   💡 Possible reasons:")
        print(f"      - Model accuracy too high ({baseline_acc:.3f}) - task too easy")
        print(f"      - Model predictions too extreme/confident")
        print(f"      - Model overfitting to noise rather than features")
        
        # Show top features anyway
        print(f"   📊 Top 5 features anyway:")
        for rank, (feature_name, importance) in enumerate(feature_importance[:5]):
            print(f"      {rank+1}. {feature_name}: {importance:+.6f}")
    else:
        print(f"   📊 TOP {min(top_k, len(meaningful_features))} IMPORTANT FEATURES:")
        for rank, (feature_name, importance) in enumerate(meaningful_features[:top_k]):
            print(f"      {rank+1:2d}. {feature_name}: {importance:+.4f} accuracy drop")
    
    return feature_importance

def validate_groups(groups, label):
    """Validate group structure and detect potential issues"""
    unique_groups = set(groups)
    group_counts = {g: np.sum(groups == g) for g in unique_groups}
    
    print(f"\n🔍 {label} Group Validation:")
    print(f"   Unique groups: {len(unique_groups)}")
    print(f"   Total samples: {len(groups)}")
    print(f"   Sample groups: {list(unique_groups)[:5]}...")
    print(f"   Group sizes: min={min(group_counts.values())}, max={max(group_counts.values())}, avg={np.mean(list(group_counts.values())):.1f}")
    
    # Check for proper group naming
    ecg_groups = [g for g in unique_groups if 'ecg' in g]
    eeg_groups = [g for g in unique_groups if 'eeg' in g]
    
    if ecg_groups:
        print(f"   ECG groups: {len(ecg_groups)} (e.g., {ecg_groups[0]})")
    if eeg_groups:
        print(f"   EEG groups: {len(eeg_groups)} (e.g., {eeg_groups[0]})")
    
    # Warning checks
    if len(unique_groups) < 10:
        print(f"   ⚠️  WARNING: Very few unique groups detected!")
    
    # Check for artificial grouping patterns
    artificial_patterns = ['_000', '_001', '_002']
    artificial_groups = [g for g in unique_groups if any(pattern in g for pattern in artificial_patterns)]
    if artificial_groups:
        print(f"   🚨 ARTIFICIAL GROUPING DETECTED: {artificial_groups[:3]}...")
        print(f"   💡 This suggests fake groups instead of real patient/subject IDs")
    else:
        print(f"   ✅ Group naming looks legitimate (no artificial patterns)")

def main(model_to_train='all'):
    import argparse
    
    # Parse command line arguments with new normalization options
    parser = argparse.ArgumentParser(description='Train ECG vs EEG classification models with enhanced features')
    parser.add_argument('model', nargs='?', default=model_to_train, 
                       help='Model to train: all, svm, simple_cnn, cnn_lstm, mlp, tcn, dual_branch')

    # Normalization options
    parser.add_argument('--normalization', choices=['smart', 'zscore', 'minmax', 'per_sample'], 
                       default='smart', help='Normalization method (default: smart - auto-detects existing normalization)')
    parser.add_argument('--norm-strategy', choices=['combined', 'separate', 'global'], 
                   default='global', help='Normalization strategy (default: global for deployment)')
        
    # Memory monitoring option
    parser.add_argument('--memory-limit', type=float, default=3.5,
                       help='Memory limit in GB for dataset processing (default: 3.5GB for GTX 1050)')
        
    # Validation strategy options
    parser.add_argument('--validation-strategy', choices=['auto', 'split', 'kfold'], 
                       default='auto', help='Validation strategy (default: auto)')
    
    # Learning curve options
    parser.add_argument('--learning-curve', action='store_true',
                       help='Run learning curve analysis before training')
    parser.add_argument('--lc-models', nargs='+', default=['mlp', 'simple_cnn', 'svm'],
                       help='Models to analyze in learning curve (default: mlp simple_cnn svm)')
    parser.add_argument('--lc-fractions', nargs='+', type=float, 
                       default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                       help='Sample fractions for learning curve analysis')
    parser.add_argument('--lc-folds', type=int, default=3,
                       help='Number of CV folds for learning curve (default: 3)')
    parser.add_argument('--quick-lc', action='store_true',
                       help='Run quick learning curve analysis (fewer models, fractions, folds)')
    
    # Dataset options
    parser.add_argument('--reload', action='store_true',
                       help='Force reload dataset (ignore cache)')
    parser.add_argument('--dataset-fraction', type=float, default=1.0,
                   help='Fraction of dataset to use (0.1 = 10%%, 0.5 = 50%%, 1.0 = 100%% - default)')
    
    # Enhanced options
    parser.add_argument('--memory-monitoring', action='store_true', default=True,
                       help='Enable detailed memory monitoring (default: True)')
    parser.add_argument('--progress-bars', action='store_true', default=True,
                       help='Enable progress bars (default: True)')
    
    # Group-based validation option
    parser.add_argument('--use-groups', action='store_true', default=True,
                       help='Use group-based CV to prevent data leakage (default: True)')
    
    # Data source options - NEW
    parser.add_argument('--ecg-path', default='data/mit-bih',
                       help='Path to MIT-BIH ECG data directory (default: data/mit-bih)')
    parser.add_argument('--eeg-path', default='data/bdf', 
                       help='Path to DEAP EEG BDF files directory (default: data/bdf)')
    
    # Parse arguments from command line
    args = parser.parse_args()
    
    print("🚀 ENHANCED TRAINING PIPELINE")
    print("=" * 70)
    print(f"🔧 Normalization: {args.normalization} (strategy: {args.norm_strategy})")
    if args.normalization == 'smart':
        print("   💡 Smart mode will auto-detect existing normalization and prevent double normalization")
    print(f"📊 Dataset fraction: {args.dataset_fraction*100:.1f}%")
    print(f"💾 Memory limit: {args.memory_limit}GB")
    print(f"🎯 Validation strategy: {args.validation_strategy}")
    print(f"🔒 Group-based CV: {'Enabled' if args.use_groups else 'Disabled'}")
    print(f"📊 Memory monitoring: {'Enabled' if args.memory_monitoring else 'Disabled'}")
    print(f"📈 Progress bars: {'Enabled' if args.progress_bars else 'Disabled'}")
    print(f"🖥️  GPU optimization: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")
    print(f"📂 ECG data path: {args.ecg_path}")
    print(f"📂 EEG data path: {args.eeg_path}")

    # Initial memory status
    print("\n📊 Initial system status:")
    import psutil
    initial_memory = psutil.Process().memory_info().rss / 1024**3
    print(f"💾 Initial memory usage: {initial_memory:.2f}GB")
    
    print("\n📦 Preparing dataset with smart normalization...")

    # Enhanced dataset loading with direct file processing
    dataset_start_time = time.time()
    X_train, X_test, y_train, y_test, metadata = prepare_dataset(
        args.ecg_path, args.eeg_path,  # Use direct paths instead of CSV files
        normalization=args.normalization,
        normalization_strategy=args.norm_strategy,
        validate_alignment=True,
        force_reload=args.reload,
        memory_limit_gb=args.memory_limit,
        dataset_fraction=args.dataset_fraction
    )
    dataset_load_time = time.time() - dataset_start_time
    current_memory = psutil.Process().memory_info().rss / 1024**3

    print("✅ Dataset ready!")
    print(f"   📊 Dataset loaded in {dataset_load_time:.1f}s")
    print(f"   🚀 Train: {X_train.shape}, Test: {X_test.shape}")
    if args.dataset_fraction < 1.0:
        print(f"   📉 Using {args.dataset_fraction*100:.1f}% of full dataset")
    print(f"   📈 Total features: {X_train.shape[1]:,}")
    print(f"   🧠 EEG structure: {metadata.get('eeg_channels', 'unknown')} channels × {metadata.get('eeg_timepoints', 'unknown')} timepoints")
    print(f"   💾 Memory usage: {current_memory:.2f}GB (Δ{current_memory-initial_memory:+.2f}GB)")
    print(f"   🔢 Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   📊 Data stats: mean={X_train.mean():.6f}, std={X_train.std():.6f}")
    
    # CRITICAL: Validate groups for proper CV
    groups_train = None
    if args.use_groups and 'groups_train' in metadata:
        groups_train = metadata['groups_train']
        
        # NEW: Validate group structure
        validate_groups(groups_train, "Training")
        
        unique_groups = len(set(groups_train))
        print(f"\n🔒 Group-based CV Analysis:")
        print(f"   📊 Total unique groups: {unique_groups}")
        
        # Validate sufficient groups for CV
        if unique_groups < RUNTIME_CONFIG['global']['n_splits']:
            print(f"   ⚠️  Warning: Only {unique_groups} groups for {RUNTIME_CONFIG['global']['n_splits']} folds!")
            print(f"   💡 Consider reducing n_splits or disabling groups with --no-use-groups")
            
            # Ask user what to do
            if unique_groups >= 3:
                response = input(f"   🤔 Continue with {unique_groups} groups? Reduce folds to {unique_groups}? [y/n/r]: ").lower()
                if response == 'r':
                    RUNTIME_CONFIG['global']['n_splits'] = unique_groups
                    print(f"   🔧 Reduced CV folds to {unique_groups}")
                elif response != 'y':
                    print("   🛑 Training stopped.")
                    return
            else:
                print(f"   🚨 Too few groups ({unique_groups}) for reliable CV. Consider:")
                print(f"      - Adding more data files")
                print(f"      - Using --validation-strategy split")
                print(f"      - Disabling groups with --no-use-groups")
                return
        
        # Check for balanced representation
        ecg_groups = [g for g in groups_train if 'ecg' in g]
        eeg_groups = [g for g in groups_train if 'eeg' in g]
        ecg_unique = len(set(ecg_groups))
        eeg_unique = len(set(eeg_groups))
        
        print(f"   ❤️  ECG: {ecg_unique} unique patients")
        print(f"   🧠 EEG: {eeg_unique} unique subjects")
        
        if ecg_unique < 5 or eeg_unique < 5:
            print(f"   ⚠️  Warning: Few unique groups per signal type!")
            print(f"   💡 This may lead to unstable CV results")
        
    elif args.use_groups:
        print(f"   ⚠️  Group-based CV requested but no groups found in metadata")
        print(f"   🔄 Falling back to standard StratifiedKFold")
    
    # Check for extreme values
    extreme_count = np.sum(np.abs(X_train) > 6)
    if extreme_count > 0:
        extreme_percent = extreme_count / X_train.size * 100
        print(f"   ⚠️  Warning: {extreme_count:,} extreme values (>6 std, {extreme_percent:.3f}%)")
        if extreme_percent > 1.0:
            print("   💡 Consider using --normalization per_sample for safer training")
            response = input("   🤔 Continue anyway? [y/N]: ").lower()
            if response not in ['y', 'yes']:
                print("   🛑 Training stopped. Try: python train.py --normalization per_sample")
                return
    else:
        print(f"   ✅ No extreme outliers detected - normalization looks good!")

    # Learning Curve Analysis (if requested)
    if args.learning_curve:
        from learning_curve import run_learning_curve_analysis
        
        print("\n🎯 LEARNING CURVE ANALYSIS REQUESTED")
        print("=" * 60)
        
        if args.quick_lc:
            print("⚡ Running QUICK learning curve analysis...")
            lc_models = ['mlp', 'svm']  # Fastest models
            lc_fractions = [0.05, 0.1, 0.2, 0.5]  # Fewer sample sizes
            lc_folds = 2  # Fewer folds
        else:
            lc_models = args.lc_models
            lc_fractions = args.lc_fractions
            lc_folds = args.lc_folds
        
        print(f"📊 Analyzing models: {', '.join(lc_models)}")
        print(f"📈 Sample fractions: {lc_fractions}")
        print(f"🔄 CV folds: {lc_folds}")
        
        try:
            results, recommendations = run_learning_curve_analysis(
                X_train, y_train, X_test, y_test,
                models_to_analyze=lc_models,
                sample_fractions=lc_fractions,
                n_folds=lc_folds
            )
            
            # Memory cleanup after learning curve analysis
            print("🧹 Cleaning up memory after learning curve analysis...")
            cleanup_memory()
            
            if recommendations:
                print("\n💡 LEARNING CURVE INSIGHTS:")
                for model, rec in recommendations.items():
                    if rec['efficient_fraction'] < 0.5:
                        estimated_time_savings = (1 - rec['efficient_fraction']) * 100
                        print(f"   🚀 {model}: Can achieve {rec['best_val_accuracy']:.3f} accuracy")
                        print(f"      with only {rec['efficient_samples']:,} samples ({rec['efficient_fraction']*100:.1f}%)")
                        print(f"      Potential time savings: ~{estimated_time_savings:.1f}%")
                
                # Ask user if they want to proceed with recommended dataset sizes
                if any(rec['efficient_fraction'] < 0.8 for rec in recommendations.values()):
                    response = input("\n🤔 Learning curve suggests smaller datasets may be sufficient. "
                                   "Continue with full training? [y/N/s(mall)]: ").lower()
                    
                    if response == 's' or response == 'small':
                        print("📉 Proceeding with recommended smaller dataset sizes...")
                        # Update dataset size based on recommendations
                        best_model = min(recommendations.items(), 
                                       key=lambda x: x[1]['efficient_samples'])
                        recommended_size = best_model[1]['efficient_samples']
                        
                        # Subsample training data
                        from sklearn.model_selection import train_test_split
                        if recommended_size < len(X_train):
                            X_train, _, y_train, _ = train_test_split(
                                X_train, y_train,
                                train_size=recommended_size,
                                stratify=y_train,
                                random_state=42
                            )
                            print(f"📊 Using {len(X_train):,} samples for training "
                                  f"(reduced from {len(X_train) + len(_):,})")
                    
                    elif response != 'y' and response != 'yes':
                        print("🛑 Training cancelled based on learning curve analysis.")
                        print("💡 Consider using the --quick-lc flag for faster analysis next time.")
                        return
            
        except ImportError:
            print("❌ learning_curve.py not found. Please ensure it's in the same directory.")
            print("⏭️  Proceeding with regular training...")
        except Exception as e:
            print(f"❌ Error during learning curve analysis: {str(e)}")
            print("⏭️  Proceeding with regular training...")

    # Enhanced Training Pipeline
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)

    models_to_train = []
    if args.model == 'all':
        models_to_train = ['svm', 'simple_cnn', 'cnn_lstm', 'mlp', 'tcn', 'dual_branch']
    elif args.model in ['svm', 'simple_cnn', 'cnn_lstm', 'mlp', 'tcn', 'dual_branch']:
        models_to_train = [args.model]
    else:
        print(f"❌ Unknown model: '{args.model}'")
        print(f"Available models: svm, simple_cnn, cnn_lstm, mlp, tcn, dual_branch")
        print("💡 Use 'all' to train all models")
        return

    # Training with memory management
    print(f"\n🚀 Training {len(models_to_train)} models with memory leak prevention...")
    print("=" * 70)
    
    training_results = {}
    overall_start_time = time.time()
    
    for i, model_name in enumerate(models_to_train):
        model_start_time = time.time()
        
        print(f"\n{'='*20} {model_name.upper()} ({i+1}/{len(models_to_train)}) {'='*20}")
        
        # Memory monitoring before training
        mem_before = psutil.Process().memory_info().rss / 1024**3
        print(f"📊 Memory before {model_name}: {mem_before:.2f}GB")
        
        try:
            if model_name == 'svm':
                train_svm(X_train, y_train, X_test, y_test, output_dir)
            elif model_name == 'simple_cnn':
                train_keras_model(build_simple_cnn, X_train, y_train, X_test, y_test, 
                                output_dir, 'simple_cnn', RUNTIME_CONFIG['simple_cnn'], 
                                metadata, args.validation_strategy, groups_train)
            elif model_name == 'cnn_lstm':
                train_keras_model(build_cnn_lstm, X_train, y_train, X_test, y_test, 
                                output_dir, 'cnn_lstm', RUNTIME_CONFIG['cnn_lstm'], 
                                metadata, args.validation_strategy, groups_train)
            elif model_name == 'mlp':
                train_keras_model(build_mlp, X_train, y_train, X_test, y_test, 
                                output_dir, 'mlp', RUNTIME_CONFIG['mlp'], 
                                metadata, args.validation_strategy, groups_train)
            elif model_name == 'tcn':
                train_keras_model(build_tcn, X_train, y_train, X_test, y_test, 
                                output_dir, 'tcn', RUNTIME_CONFIG['tcn'], 
                                metadata, args.validation_strategy, groups_train)
            elif model_name == 'dual_branch':
                train_dual_branch(X_train, y_train, X_test, y_test, output_dir, metadata, args.validation_strategy, groups_train)
            
            # Record success
            model_time = time.time() - model_start_time
            training_results[model_name] = {
                'training_time': model_time,
                'status': 'success'
            }
            
            print(f"✅ {model_name} completed in {model_time:.1f}s")

            # Feature importance analysis
            print(f"🔍 Running feature importance analysis for {model_name}...")
            try:
                if model_name == 'svm':
                    # Load the saved SVM model
                    import joblib
                    svm_model_path = os.path.join(output_dir, 'svm_model.joblib')
                    if os.path.exists(svm_model_path):
                        svm_model = joblib.load(svm_model_path)
                        importance_scores = analyze_feature_importance(svm_model, X_test, y_test, model_name)
                    else:
                        print(f"   ⚠️  SVM model file not found, skipping importance analysis")
                        importance_scores = []
                else:
                    # Load the saved Keras model
                    from tensorflow.keras.models import load_model
                    keras_model_path = os.path.join(output_dir, f'{model_name}_final.keras')
                    if os.path.exists(keras_model_path):
                        keras_model = load_model(keras_model_path)
                        importance_scores = analyze_feature_importance(keras_model, X_test, y_test, model_name)
                        # Clean up model from memory
                        del keras_model
                    else:
                        print(f"   ⚠️  Keras model file not found, skipping importance analysis")
                        importance_scores = []
                
                # Save importance scores
                if importance_scores:
                    import pickle
                    importance_path = os.path.join(output_dir, f'{model_name}_feature_importance.pkl')
                    with open(importance_path, 'wb') as f:
                        pickle.dump(importance_scores, f)
                    print(f"   💾 Feature importance saved to: {importance_path}")
                
            except Exception as importance_error:
                print(f"   ⚠️  Feature importance analysis failed: {importance_error}")
            
        except Exception as e:
            model_time = time.time() - model_start_time
            training_results[model_name] = {
                'training_time': model_time,
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ {model_name} failed after {model_time:.1f}s: {str(e)}")
        
        finally:
            # Memory cleanup after each model (this prevents the memory leak!)
            print("🧹 Cleaning up memory...")
            cleanup_memory()
            
            # Memory monitoring after cleanup
            mem_after = psutil.Process().memory_info().rss / 1024**3
            print(f"📊 Memory after cleanup: {mem_after:.2f}GB (Δ{mem_after-mem_before:+.2f}GB)")
            
            # Force garbage collection
            gc.collect()

    # Final summary
    total_time = time.time() - overall_start_time
    print(f"\n🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Results summary:")
    
    successful_models = [m for m, r in training_results.items() if r['status'] == 'success']
    failed_models = [m for m, r in training_results.items() if r['status'] == 'failed']
    
    print(f"   ✅ Successful: {len(successful_models)}/{len(models_to_train)}")
    print(f"   ❌ Failed: {len(failed_models)}/{len(models_to_train)}")
    
    if successful_models:
        fastest_model = min(training_results.items(), key=lambda x: x[1]['training_time'])
        print(f"   🚀 Fastest: {fastest_model[0]} ({fastest_model[1]['training_time']:.1f}s)")
    
    if failed_models:
        print(f"   💥 Failed models: {', '.join(failed_models)}")
    
    # Final memory status
    final_memory = psutil.Process().memory_info().rss / 1024**3
    print(f"📊 Final memory usage: {final_memory:.2f}GB (total change: {final_memory-initial_memory:+.2f}GB)")
    
    print("\n💡 Next steps:")
    print("   • Check model performance in the models/ directory")
    print("   • Compare training curves to identify best performing models")
    print("   • Use the best performing model for inference")
    if args.learning_curve:
        print("   • Review learning curve analysis for dataset optimization insights")
    
    # Group validation summary
    if groups_train:
        print(f"   🔒 Group-based CV used: {len(set(groups_train))} unique groups ensured no data leakage")

if __name__ == '__main__':
    import sys
    
    # Handle both old and new calling conventions
    if len(sys.argv) == 1:
        # No arguments - run with defaults
        main('all')
    elif len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
        # Single model argument (legacy support)
        main(sys.argv[1])
    else:
        # New argument parsing
        main()