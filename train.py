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
    RUNTIME_CONFIG['global']['epochs'] = 30  # Reduced from 50
    RUNTIME_CONFIG['global']['n_splits'] = 3  # Reduced from 5
    
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
        
        # # Model checkpointing
        # ModelCheckpoint(
        #     os.path.join(output_dir, f'{model_name}_best.keras'), 
        #     save_best_only=True, 
        #     monitor='val_accuracy',
        #     verbose=0,
        #     save_format='tf'
        # ),
        ModelCheckpoint(
            filepath=os.path.join(output_dir, f"{model_name}_best"),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=0,
            save_format='tf'  # Use TensorFlow format for compatibility
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

    # Evaluate on test set
    print("🔍 Evaluating on test set...")
    y_probs = model.predict(X_test_model, verbose=0)
    y_pred = (y_probs > 0.5).astype(int).flatten() if num_classes == 2 else np.argmax(y_probs, axis=1)
    print_evaluation(y_test, y_pred, name)
    
    # Memory monitoring after training
    monitor_memory_usage(f"after {name}")
    
    print(f"✅ {name} model saved")
    return model

def train_with_kfold(model_builder, X_train, y_train, X_test, y_test, output_dir, name, config, metadata):
    """Train model using K-fold cross-validation with enhanced features"""
    print(f"🚀 Training {name} with {RUNTIME_CONFIG['global']['n_splits']}-fold cross-validation...")
    
    # Memory monitoring
    monitor_memory_usage(f"before {name} k-fold")
    
    # Prepare data with correct shape for this model type
    X_train_model, X_test_model, input_shape = prepare_model_data(X_train, X_test, name, metadata)
    
    num_classes = len(np.unique(y_train))
    loss_fn = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

    # UPDATED: Use centralized class weight computation
    class_weight_dict = compute_class_weights(y_train)

    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=RUNTIME_CONFIG['global']['n_splits'], shuffle=True, random_state=RUNTIME_CONFIG['global']['random_state'])
    fold_scores = []

    # Progress bar for folds
    fold_progress = tqdm(
        total=RUNTIME_CONFIG['global']['n_splits'], 
        desc=f"🔄 {name} K-Fold", 
        position=0,
        leave=True
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_model, y_train)):
        fold_progress.set_description(f"🔄 {name} Fold {fold + 1}")
        
        X_tr, X_val = X_train_model[train_idx], X_train_model[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

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
    y_pred = (y_probs > 0.5).astype(int).flatten() if num_classes == 2 else np.argmax(y_probs, axis=1)
    print_evaluation(y_test, y_pred, name)
    
    # Memory monitoring after training
    monitor_memory_usage(f"after {name} k-fold")
    
    print(f"✅ {name} model saved")
    return final_model

def train_keras_model(model_builder, X_train, y_train, X_test, y_test, output_dir, name, config, metadata, validation_strategy='auto'):
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
        return train_with_kfold(model_builder, X_train, y_train, X_test, y_test, output_dir, name, config, metadata)
    else:
        return train_with_split(model_builder, X_train, y_train, X_test, y_test, output_dir, name, config, metadata)

def train_dual_branch(X_train, y_train, X_test, y_test, output_dir, metadata, validation_strategy='auto'):
    """
    Train Dual-Branch CNN with proper data preparation and enhanced features
    Kept as separate function as requested
    """
    print("\n🚀 Training Dual-Branch CNN...")

    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ECG', 'EEG'], yticklabels=['ECG', 'EEG'])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        filepath = os.path.join(output_dir, 'dual_branch_confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"📉 Confusion matrix saved to: {filepath}")

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
        skf = StratifiedKFold(n_splits=RUNTIME_CONFIG['global']['n_splits'], shuffle=True, random_state=RUNTIME_CONFIG['global']['random_state'])
        fold_scores = []
        
        # Progress bar for folds
        fold_progress = tqdm(
            total=RUNTIME_CONFIG['global']['n_splits'], 
            desc="🔄 Dual-Branch K-Fold", 
            position=0,
            leave=True
        )
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_model, y_train)):
            fold_progress.set_description(f"🔄 Dual-Branch Fold {fold + 1}")
            
            X_tr, X_val = X_train_model[train_idx], X_train_model[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = build_dual_branch_cnn(input_shape=input_shape, num_classes=2, config=RUNTIME_CONFIG['dual_branch'])
            
            # UPDATED: Use centralized optimizer creation
            optimizer = create_optimizer(RUNTIME_CONFIG['dual_branch']['learning_rate'])
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Enhanced callbacks for this fold
            callbacks = create_enhanced_callbacks(f"dual_branch_fold{fold+1}", output_dir, progress_position=1)

            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                epochs=RUNTIME_CONFIG['global']['epochs'],
                batch_size=RUNTIME_CONFIG['dual_branch']['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
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

        print(f"\n✅ Dual-Branch CNN K-Fold Results:")
        print(f"   Mean CV Score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        log_fold_metrics_to_csv('dual_branch', fold_scores, output_dir)

        # Train final model
        print(f"\nTraining final dual-branch model on full training set...")
        final_model = build_dual_branch_cnn(input_shape=input_shape, num_classes=2, config=RUNTIME_CONFIG['dual_branch'])
        
        # UPDATED: Use centralized optimizer creation
        optimizer = create_optimizer(RUNTIME_CONFIG['dual_branch']['learning_rate'])
        final_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Enhanced callbacks for final training
        callbacks = create_enhanced_callbacks("dual_branch_final", output_dir, progress_position=0)

        final_history = final_model.fit(
            X_train_model, y_train,
            validation_split=0.1,
            epochs=RUNTIME_CONFIG['global']['epochs'],
            batch_size=RUNTIME_CONFIG['dual_branch']['batch_size'],
            callbacks=callbacks,
            verbose=0
        )

        save_training_curves(final_history, 'dual_branch', output_dir)
        y_pred = (final_model.predict(X_test_model, verbose=0) > 0.5).astype(int).flatten()
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["ECG", "EEG"]))
        plot_confusion_matrix(y_test, y_pred)

        final_model.save(os.path.join(output_dir, 'dual_branch_final.keras'))

    else:  # Simple split
        model = build_dual_branch_cnn(input_shape=input_shape, num_classes=2, config=RUNTIME_CONFIG['dual_branch'])
        
        # UPDATED: Use centralized optimizer creation
        optimizer = create_optimizer(RUNTIME_CONFIG['dual_branch']['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        print(f"📊 Model summary for dual_branch:")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Input shape: {input_shape}")

        # Enhanced callbacks
        callbacks = create_enhanced_callbacks("dual_branch", output_dir, progress_position=0)

        start_time = time.time()
        history = model.fit(
            X_train_model, y_train,
            validation_split=0.1,
            epochs=RUNTIME_CONFIG['global']['epochs'],
            batch_size=RUNTIME_CONFIG['dual_branch']['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        
        training_time = time.time() - start_time
        print(f"⏱️  Training completed in {training_time:.1f}s")

        save_training_curves(history, 'dual_branch', output_dir)
        y_pred = (model.predict(X_test_model, verbose=0) > 0.5).astype(int).flatten()
        print("📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["ECG", "EEG"]))
        plot_confusion_matrix(y_test, y_pred)

        model.save(os.path.join(output_dir, 'dual_branch_final.keras'))

    # Memory monitoring after training
    monitor_memory_usage("after dual_branch")
    print("✅ Dual-Branch CNN training complete.")

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

def main(model_to_train='all'):
    import argparse
    
    # Parse command line arguments with new normalization options
    parser = argparse.ArgumentParser(description='Train ECG vs EEG classification models with enhanced features')
    parser.add_argument('model', nargs='?', default=model_to_train, 
                       help='Model to train: all, svm, simple_cnn, cnn_lstm, mlp, tcn, dual_branch')

    # Normalization options
    parser.add_argument('--normalization', choices=['smart', 'zscore', 'minmax', 'per_sample'], 
                       default='smart', help='Normalization method (default: smart - auto-detects existing normalization)')
    parser.add_argument('--norm-strategy', choices=['combined', 'separate'], 
                       default='separate', help='Normalization strategy (default: separate)')
        
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
    print(f"📊 Memory monitoring: {'Enabled' if args.memory_monitoring else 'Disabled'}")
    print(f"📈 Progress bars: {'Enabled' if args.progress_bars else 'Disabled'}")
    print(f"🖥️  GPU optimization: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")

    # Initial memory status
    print("\n📊 Initial system status:")
    import psutil
    initial_memory = psutil.Process().memory_info().rss / 1024**3
    print(f"💾 Initial memory usage: {initial_memory:.2f}GB")
    
    print("\n📦 Preparing dataset with smart normalization...")
    ecg_csv = os.path.join('data', 'mitbih_train.csv')
    eeg_csv = os.path.join('data', 'eeg_dataset_32.csv')

    # Load dataset with enhanced monitoring
    dataset_start_time = time.time()
    X_train, X_test, y_train, y_test, metadata = prepare_dataset(
        ecg_csv, eeg_csv,
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
            
            # ADD: Memory cleanup after learning curve analysis
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
                                metadata, args.validation_strategy)
            elif model_name == 'cnn_lstm':
                train_keras_model(build_cnn_lstm, X_train, y_train, X_test, y_test, 
                                output_dir, 'cnn_lstm', RUNTIME_CONFIG['cnn_lstm'], 
                                metadata, args.validation_strategy)
            elif model_name == 'mlp':
                train_keras_model(build_mlp, X_train, y_train, X_test, y_test, 
                                output_dir, 'mlp', RUNTIME_CONFIG['mlp'], 
                                metadata, args.validation_strategy)
            elif model_name == 'tcn':
                train_keras_model(build_tcn, X_train, y_train, X_test, y_test, 
                                output_dir, 'tcn', RUNTIME_CONFIG['tcn'], 
                                metadata, args.validation_strategy)
            elif model_name == 'dual_branch':
                train_dual_branch(X_train, y_train, X_test, y_test, output_dir, metadata, args.validation_strategy)
            
            # Record success
            model_time = time.time() - model_start_time
            training_results[model_name] = {
                'training_time': model_time,
                'status': 'success'
            }
            
            print(f"✅ {model_name} completed in {model_time:.1f}s")
            
        except Exception as e:
            model_time = time.time() - model_start_time
            training_results[model_name] = {
                'training_time': model_time,
                'status': 'failed',
                'error': str(e)
            }
            print(f"❌ {model_name} failed after {model_time:.1f}s: {str(e)}")
        
        finally:
            # CRITICAL: Memory cleanup after each model (this prevents the memory leak!)
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