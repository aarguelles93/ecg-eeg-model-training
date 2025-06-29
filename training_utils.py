"""
Training utilities for ECG vs EEG classification
SAFE IMPLEMENTATION - No module-level TensorFlow imports
All TF imports happen inside functions AFTER environment variables are set

This module is designed to be imported BEFORE TensorFlow while preserving
the critical CUDA environment variable setup in train.py
"""

import numpy as np
import os
import gc
import time
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

# ===== NO TENSORFLOW IMPORT AT MODULE LEVEL =====
# This is CRITICAL - TF import must happen after environment variables are set
# import tensorflow as tf  # ❌ DON'T DO THIS!


class OptimizerFactory:
    """Centralized optimizer creation with TensorFlow version compatibility"""
    
    @staticmethod
    def create_optimizer(learning_rate=1e-4, optimizer_type='adam', **kwargs):
        """
        Create optimizers with automatic fallback for TF compatibility
        TF imported inside function to respect environment variable timing
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
            **kwargs: Additional optimizer parameters
        
        Returns:
            TensorFlow optimizer instance
        """
        # Import TF inside function - AFTER environment variables are set
        import tensorflow as tf

        print(f"🔧 Creating {optimizer_type} optimizer with lr={learning_rate}")

        if optimizer_type.lower() == 'adam':
            try:
                # Use legacy Adam explicitly for TF v1 graph mode compatibility
                from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
                return LegacyAdam(
                    learning_rate=learning_rate,
                    clipnorm=kwargs.get('clipnorm', 1.0),
                    beta_1=kwargs.get('beta_1', 0.9),
                    beta_2=kwargs.get('beta_2', 0.999),
                    epsilon=kwargs.get('epsilon', 1e-7)
                )
            except Exception as e:
                print(f"   ⚠️  Could not import legacy Adam: {e}")
                print("   ⚠️  Falling back to standard Adam (may fail in graph mode)")
                return tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    clipnorm=kwargs.get('clipnorm', 1.0),
                    beta_1=kwargs.get('beta_1', 0.9),
                    beta_2=kwargs.get('beta_2', 0.999),
                    epsilon=kwargs.get('epsilon', 1e-7)
                )

        elif optimizer_type.lower() == 'adamw':
            return tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=kwargs.get('weight_decay', 1e-4),
                clipnorm=kwargs.get('clipnorm', 1.0)
            )

        elif optimizer_type.lower() == 'sgd':
            return tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=kwargs.get('momentum', 0.9),
                clipnorm=kwargs.get('clipnorm', 1.0)
            )

        else:
            # Default to legacy Adam as a safe fallback
            return OptimizerFactory.create_optimizer(learning_rate, 'adam', **kwargs)



class CallbackFactory:
    """Centralized callback creation for training"""
    
    @staticmethod
    def create_standard_callbacks(model_name, output_dir, monitor='val_accuracy', 
                                 patience=15, reduce_lr_patience=None):
        """
        Create standard training callbacks
        TF imported inside function to respect environment variable timing
        
        Args:
            model_name: Name of model for saving
            output_dir: Directory to save model checkpoints
            monitor: Metric to monitor for callbacks
            patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction (default: patience//3)
        
        Returns:
            List of Keras callbacks
        """
        # Import TF inside function - AFTER environment variables are set
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        
        if reduce_lr_patience is None:
            reduce_lr_patience = max(3, patience // 3)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1,
                mode='max' if 'acc' in monitor else 'min'
            ),
            ModelCheckpoint(
                os.path.join(output_dir, f'{model_name}_best.keras'),
                monitor=monitor,
                save_best_only=True,
                verbose=1,
                mode='max' if 'acc' in monitor else 'min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',  # Always monitor loss for LR reduction
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            )
        ]
        
        return callbacks
    
    @staticmethod
    def create_memory_efficient_callbacks(model_name, output_dir, monitor='val_accuracy'):
        """Create callbacks optimized for limited memory scenarios"""
        # Import TF inside function
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # More aggressive early stopping and LR reduction for memory efficiency
        return [
            EarlyStopping(
                monitor=monitor,
                patience=10,  # Reduced patience
                restore_best_weights=True,
                min_delta=0.002,  # Slightly higher threshold
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,  # More aggressive reduction
                patience=3,   # Faster reduction
                min_lr=1e-6,
                verbose=1
            )
        ]


class TrainingHelper:
    """Common training utilities and helpers"""
    
    @staticmethod
    def compute_class_weights(y_train):
        """
        Compute balanced class weights for imbalanced datasets
        No TF dependency - safe to import
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        return dict(enumerate(class_weights))
    
    @staticmethod
    def compile_model(model, learning_rate, loss='binary_crossentropy', 
                     metrics=['accuracy'], optimizer_type='adam'):
        """
        Standardized model compilation
        TF imported inside function to respect environment variable timing
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            loss: Loss function
            metrics: List of metrics to track
            optimizer_type: Type of optimizer to use
            
        Returns:
            Compiled model
        """
        optimizer = OptimizerFactory.create_optimizer(learning_rate, optimizer_type)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    @staticmethod
    def prepare_labels(y, num_classes):
        """
        Prepare labels for training (binary vs categorical)
        TF imported inside function when needed
        
        Args:
            y: Raw labels
            num_classes: Number of classes
            
        Returns:
            Processed labels
        """
        if num_classes == 2:
            return y
        else:
            # Import only when needed
            import tensorflow as tf
            from tensorflow.keras.utils import to_categorical
            return to_categorical(y, num_classes)
    
    @staticmethod
    def determine_loss_and_metrics(num_classes):
        """
        Determine appropriate loss function and metrics based on problem type
        No TF dependency - safe to import
        
        Args:
            num_classes: Number of classes
            
        Returns:
            Tuple of (loss_function, metrics_list)
        """
        if num_classes == 2:
            return 'binary_crossentropy', ['accuracy']
        else:
            return 'categorical_crossentropy', ['accuracy']
    
    @staticmethod
    def safe_evaluate_model(model, X_test, y_test, verbose=0):
        """
        Safely evaluate model with error handling
        Model is already created, so TF is already imported
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            verbose: Verbosity level
            
        Returns:
            Tuple of (loss, accuracy) or (None, None) if evaluation fails
        """
        try:
            results = model.evaluate(X_test, y_test, verbose=verbose)
            if isinstance(results, list):
                return results[0], results[1]  # loss, accuracy
            else:
                return results, None
        except Exception as e:
            print(f"⚠️  Model evaluation failed: {e}")
            return None, None


class MemoryManager:
    """Memory management utilities - TF imported inside functions when needed"""
    
    @staticmethod
    def cleanup_tensorflow():
        """Comprehensive TensorFlow memory cleanup"""
        print("🧹 Cleaning up TensorFlow memory...")
        
        # Import TF inside function - safe because this is called after TF usage
        import tensorflow as tf
        
        # Clear Keras session
        tf.keras.backend.clear_session()
        
        # Clear default graph
        try:
            tf.compat.v1.reset_default_graph()
        except:
            pass
        
        # Force garbage collection
        gc.collect()
        
        # GPU memory cleanup if available
        try:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.reset_memory_stats(device.name)
        except:
            pass
        
        print("✅ Memory cleanup completed")
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage statistics - no TF dependency"""
        import psutil
        
        process = psutil.Process()
        cpu_memory_gb = process.memory_info().rss / 1024**3
        
        gpu_memory_gb = 0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_gb = info.used / 1024**3
        except:
            pass
        
        return cpu_memory_gb, gpu_memory_gb
    
    @staticmethod
    def monitor(label=""):
        """Monitor and log memory usage - no TF dependency"""
        cpu_mem, gpu_mem = MemoryManager.get_memory_usage()
        print(f"📊 Memory {label}: CPU={cpu_mem:.2f}GB, GPU={gpu_mem:.2f}GB")
        return cpu_mem, gpu_mem


class ValidationStrategy:
    """Utilities for choosing and implementing validation strategies - no TF dependency"""
    
    @staticmethod
    def choose_strategy(dataset_size, time_budget='medium', force_strategy=None):
        """
        Auto-select validation strategy based on dataset size and constraints
        
        Args:
            dataset_size: Number of training samples
            time_budget: 'low', 'medium', or 'high'
            force_strategy: Force specific strategy if provided
            
        Returns:
            Validation strategy string ('split' or 'kfold')
        """
        if force_strategy:
            return force_strategy
        
        if time_budget == 'low' or dataset_size > 50000:
            return 'split'
        elif dataset_size < 5000:
            return 'kfold'  # Small datasets benefit from k-fold
        else:
            return 'split'  # Default for medium-large datasets


# ===== CONVENIENCE FUNCTIONS =====
# These are safe because the underlying functions handle TF imports properly

def create_optimizer(learning_rate=1e-4, optimizer_type='adam', **kwargs):
    """Convenience function for optimizer creation"""
    return OptimizerFactory.create_optimizer(learning_rate, optimizer_type, **kwargs)

def create_callbacks(model_name, output_dir, monitor='val_accuracy', patience=15):
    """Convenience function for callback creation"""
    return CallbackFactory.create_standard_callbacks(model_name, output_dir, monitor, patience)

def compute_class_weights(y_train):
    """Convenience function for class weight computation"""
    return TrainingHelper.compute_class_weights(y_train)

def compile_model(model, learning_rate, loss='binary_crossentropy', metrics=['accuracy']):
    """Convenience function for model compilation"""
    return TrainingHelper.compile_model(model, learning_rate, loss, metrics)

@staticmethod
def cleanup_memory():
    """
    Clean up TensorFlow and Python memory to prevent leaks (ENHANCED)
    """
    import tensorflow as tf
    import gc
    import ctypes

    print("🧹 Cleaning up TensorFlow memory...")

    # Clear Keras session
    tf.keras.backend.clear_session()

    # Reset default graph (for TF v1 compatibility)
    tf.compat.v1.reset_default_graph()

    # Garbage collect
    gc.collect()
    gc.collect()  # Call twice for safety

    # Optional: Ask OS to release memory (Linux only)
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass

    print("✅ Memory cleanup completed")

def monitor_memory(label=""):
    """Convenience function for memory monitoring"""
    return MemoryManager.monitor(label)

# ===== IMPORT SAFETY CHECK =====
def verify_import_safety():
    """
    Verify that this module can be imported safely before TF environment setup
    Returns True if safe, False if TF was already imported
    """
    import sys
    tf_modules = [name for name in sys.modules.keys() if name.startswith('tensorflow')]
    
    if tf_modules:
        print(f"⚠️  Warning: TensorFlow modules already imported: {tf_modules}")
        print("   Environment variables may not take effect!")
        return False
    else:
        print("✅ Import safety check passed - TensorFlow not yet imported")
        return True

# Run safety check when module is imported
if __name__ != "__main__":
    verify_import_safety()