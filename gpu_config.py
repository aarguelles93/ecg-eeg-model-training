"""
GPU memory configuration for systems with limited VRAM
Optimized for GTX 1050 (2GB) with large datasets
FIXED: TensorFlow import moved inside functions to respect environment variables
"""

import os
import gc
import numpy as np

# Set correct CUDA path before any TensorFlow operations
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
os.environ['CUDA_DIR'] = '/usr/lib/cuda'

# def configure_tensorflow_memory_management():
#     """
#     Configure TensorFlow to prevent memory leaks - ADD TO EXISTING gpu_config.py
#     Call this function immediately after importing TensorFlow
#     """
#     # FIXED: Import TF inside function, not at module level
#     import tensorflow as tf
    
#     print("🔧 Configuring TensorFlow memory management...")
    
#     # Disable eager execution (major memory leak source)
#     tf.compat.v1.disable_eager_execution()
    
#     # Configure session for minimal memory usage
#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.6  # Use max 60% of GPU
#     config.allow_soft_placement = True
#     config.log_device_placement = False
    
#     # Memory optimization settings
#     config.inter_op_parallelism_threads = 1
#     config.intra_op_parallelism_threads = 1
#     config.use_per_session_threads = True
    
#     # Create and set global session
#     session = tf.compat.v1.Session(config=config)
#     tf.compat.v1.keras.backend.set_session(session)
    
#     print("✅ TensorFlow memory management configured")
#     return session

def configure_tensorflow_memory_management():
    """
    Configure TensorFlow memory management with memory growth (no memory limit)
    Safe for WSL and cuDNN-compatible LSTM
    """
    import tensorflow as tf

    print("🔧 Configuring TensorFlow memory management...")

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        for gpu in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Memory growth enabled for {gpu.name}")
            except RuntimeError as e:
                print(f"⚠️  Memory growth setup failed: {e}")
    else:
        print("❌ No GPU devices found.")

    return True


# def cleanup_tensorflow_memory():
#     """
#     Aggressive memory cleanup between models - ADD TO EXISTING gpu_config.py
#     """
#     # FIXED: Import TF inside function, not at module level
#     import tensorflow as tf
    
#     print("🧹 Cleaning up TensorFlow memory...")
    
#     # Clear Keras session
#     tf.keras.backend.clear_session()
    
#     # Clear default graph
#     tf.compat.v1.reset_default_graph()
    
#     # Force garbage collection
#     gc.collect()
    
#     # GPU memory cleanup if available
#     try:
#         physical_devices = tf.config.experimental.list_physical_devices('GPU')
#         if physical_devices:
#             for device in physical_devices:
#                 tf.config.experimental.reset_memory_stats(device.name)
#     except:
#         pass
    
#     print("✅ Memory cleanup completed")

def cleanup_tensorflow_memory():
    """
    Aggressive memory cleanup between models
    """
    import tensorflow as tf

    print("🧹 Cleaning up TensorFlow memory...")

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.reset_memory_stats(device.name)
    except:
        pass

    print("✅ Memory cleanup completed")



# def configure_gpu_memory(enable_memory_growth=True, memory_limit_mb=None):
#     """
#     Configure GPU memory settings for limited VRAM scenarios
#     Optimized for GTX 1050 with 2GB VRAM
    
#     Args:
#         enable_memory_growth: Allow TensorFlow to allocate GPU memory as needed
#         memory_limit_mb: Hard limit GPU memory usage (default: 1700MB for GTX 1050)
#     """
#     # FIXED: Import TF inside function, not at module level
#     import tensorflow as tf
    
#     print("🔧 Configuring GPU memory settings for GTX 1050...")
    
#     # Set conservative memory limit for GTX 1050
#     if memory_limit_mb is None:
#         memory_limit_mb = 1700  # Conservative limit for 2GB card
    
#     # Get available GPUs
#     physical_devices = tf.config.list_physical_devices('GPU')
    
#     if not physical_devices:
#         print("❌ No GPU devices found. Running on CPU.")
#         return False
    
#     print(f"✅ Found {len(physical_devices)} GPU(s): {[device.name for device in physical_devices]}")
    
#     try:
#         # Configure each GPU
#         for gpu in physical_devices:
#             if enable_memory_growth:
#                 # Enable memory growth (recommended for limited VRAM)
#                 tf.config.experimental.set_memory_growth(gpu, True)
#                 print(f"✅ Memory growth enabled for {gpu.name}")
            
#             if memory_limit_mb:
#                 # Set hard memory limit
#                 tf.config.set_logical_device_configuration(
#                     gpu,
#                     [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
#                 )
#                 print(f"🎯 Memory limit set to {memory_limit_mb}MB for {gpu.name}")
        
#         return True
        
#     except RuntimeError as e:
#         print(f"⚠️  GPU configuration failed: {e}")
#         print("   This usually happens if TensorFlow is already initialized.")
#         print("   Make sure to call configure_gpu_memory() before any TF operations.")
#         return False

def configure_gpu_memory(enable_memory_growth=True):
    """
    Configure GPU memory settings using memory growth only (WSL-safe)
    """
    import tensorflow as tf

    print("🔧 Configuring GPU memory settings...")

    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("❌ No GPU devices found. Running on CPU.")
        return False

    print(f"✅ Found {len(physical_devices)} GPU(s): {[device.name for device in physical_devices]}")

    try:
        for gpu in physical_devices:
            if enable_memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Memory growth enabled for {gpu.name}")

        return True

    except RuntimeError as e:
        print(f"⚠️  GPU configuration failed: {e}")
        print("   Make sure to call configure_gpu_memory() before any TF operations.")
        return False

def get_optimal_batch_size(model_type='cnn', available_vram_gb=2, safety_factor=0.7):
    """
    Calculate optimal batch size based on available GPU memory
    
    Args:
        model_type: Type of model ('cnn', 'lstm', 'mlp', etc.)
        available_vram_gb: Available GPU memory in GB
        safety_factor: Safety factor to avoid OOM (0.5-0.8 recommended)
    """
    
    # Memory usage estimates per sample (MB) for different model types
    memory_per_sample = {
        'mlp': 0.1,           # Simple dense layers
        'simple_cnn': 0.5,    # Basic CNN
        'cnn_lstm': 1.0,      # CNN + LSTM (more memory intensive)
        'tcn': 0.8,           # Temporal Convolutional Network
        'dual_branch': 1.2    # Dual-branch CNN (most memory intensive)
    }
    
    base_memory = memory_per_sample.get(model_type, 0.5)
    
    # Calculate batch size
    available_mb = available_vram_gb * 1024 * safety_factor
    base_overhead = 200  # MB for TensorFlow overhead
    usable_memory = available_mb - base_overhead
    
    optimal_batch_size = int(usable_memory / base_memory)
    
    # Ensure reasonable bounds
    optimal_batch_size = max(8, min(optimal_batch_size, 256))
    
    print(f"📊 Optimal batch size for {model_type}: {optimal_batch_size}")
    print(f"   Available VRAM: {available_vram_gb:.1f}GB")
    print(f"   Safety factor: {safety_factor}")
    print(f"   Estimated memory per sample: {base_memory:.1f}MB")
    
    return optimal_batch_size

def setup_mixed_precision():
    """
    Configure precision for GTX 1050 (Pascal architecture)
    Mixed precision can be slower on compute capability < 7.0, so we use float32
    """
    # FIXED: Import TF inside function, not at module level
    import tensorflow as tf
    
    try:
        # Use float32 for GTX 1050 (Pascal architecture) - better performance
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        print("✅ Using float32 precision (optimized for GTX 1050/Pascal)")
        print("   This provides better performance than mixed precision on compute capability 6.1")
        return True
        
    except Exception as e:
        print(f"⚠️  Precision setup failed: {e}")
        return False

def monitor_gpu_memory():
    """
    Monitor GPU memory usage during training
    """
    try:
        # Get GPU memory info
        # FIXED: Import TF inside function, not at module level
        import tensorflow as tf
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            # This requires nvidia-ml-py package: pip install nvidia-ml-py
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            used_mb = info.used / 1024**2
            total_mb = info.total / 1024**2
            usage_percent = (info.used / info.total) * 100
            
            print(f"🖥️  GPU Memory: {used_mb:.0f}MB / {total_mb:.0f}MB ({usage_percent:.1f}%)")
            
            return used_mb, total_mb, usage_percent
    except ImportError:
        print("💡 Install nvidia-ml-py for GPU memory monitoring: pip install nvidia-ml-py")
    except Exception as e:
        print(f"⚠️  GPU memory monitoring failed: {e}")
    
    return None, None, None

def adaptive_batch_size_finder(model_builder, input_shape, start_batch_size=64):
    """
    Automatically find the largest batch size that fits in GPU memory
    """
    # FIXED: Import TF inside function, not at module level
    import tensorflow as tf
    
    print("🔍 Finding optimal batch size through binary search...")
    
    max_batch_size = start_batch_size
    min_batch_size = 1
    optimal_batch_size = 1
    
    while min_batch_size <= max_batch_size:
        test_batch_size = (min_batch_size + max_batch_size) // 2
        
        try:
            # Clear any existing models
            tf.keras.backend.clear_session()
            
            # Create test model
            model = model_builder(input_shape=input_shape, num_classes=2)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Create dummy data
            X_test = np.random.random((test_batch_size, *input_shape)).astype(np.float32)
            y_test = np.random.randint(0, 2, test_batch_size).astype(np.float32)
            
            # Test forward pass
            _ = model.predict(X_test, batch_size=test_batch_size, verbose=0)
            
            print(f"   ✅ Batch size {test_batch_size} works")
            optimal_batch_size = test_batch_size
            min_batch_size = test_batch_size + 1
            
        except tf.errors.ResourceExhaustedError:
            print(f"   ❌ Batch size {test_batch_size} too large (OOM)")
            max_batch_size = test_batch_size - 1
            
        except Exception as e:
            print(f"   ⚠️  Error with batch size {test_batch_size}: {e}")
            max_batch_size = test_batch_size - 1
    
    print(f"🎯 Optimal batch size found: {optimal_batch_size}")
    return optimal_batch_size

def smart_dataset_chunking(X, y, chunk_size_gb=1.5):
    """
    Split large datasets into GPU-friendly chunks
    Useful when your dataset is larger than GPU memory
    """
    data_size_gb = X.nbytes / 1024**3
    
    if data_size_gb <= chunk_size_gb:
        print(f"📊 Dataset size ({data_size_gb:.2f}GB) fits in memory")
        return [(X, y)]
    
    n_chunks = int(np.ceil(data_size_gb / chunk_size_gb))
    chunk_size = len(X) // n_chunks
    
    print(f"📦 Splitting dataset into {n_chunks} chunks of ~{chunk_size:,} samples each")
    
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < n_chunks - 1 else len(X)
        chunks.append((X[start_idx:end_idx], y[start_idx:end_idx]))
    
    return chunks

def gpu_friendly_config():
    """
    Complete GPU configuration optimized for GTX 1050 with correct CUDA paths
    FIXED: TensorFlow import moved inside function
    """
    print("🚀 Setting up GPU-friendly configuration for GTX 1050 (2GB VRAM)...")
    
    # Keep all your existing CUDA path setup
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
    os.environ['CUDA_DIR'] = '/usr/lib/cuda'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    print("✅ CUDA libdevice path set to: /usr/lib/cuda")
    
    # NOW import TensorFlow (after environment variables are set)
    import tensorflow as tf
    
    # Rest of your existing GPU configuration...
    gpu_available = configure_gpu_memory(
        enable_memory_growth=True,
        # memory_limit_mb=1700  # Conservative limit for GTX 1050
    )
    
    if not gpu_available:
        print("❌ No GPU devices found. Running on CPU.")
        return False
    
    # Configure TensorFlow memory management
    configure_tensorflow_memory_management()

    setup_mixed_precision()
    
    print("✅ GPU configuration complete! GTX 1050 optimized with correct CUDA paths.")
    return True

# Updated config for your learning curve analysis
def get_gpu_optimized_config():
    """
    Return GPU-optimized configurations for GTX 1050 (2GB VRAM)
    More conservative batch sizes and model complexity
    """
    return {
        'simple_cnn': {
            'learning_rate': 1e-4,
            'batch_size': 24,  # Conservative for 2GB VRAM
            'dropout': 0.5,
            'filters': [16, 32],  # Reduced complexity for memory
            'kernel_sizes': [3, 3]
        },
        'cnn_lstm': {
            'learning_rate': 2e-4,  # INCREASED for faster convergence
            'batch_size': 32,       # INCREASED from 12 for efficiency
            'dropout': 0.5,         # INCREASED to compensate for faster learning
            'filters': [16, 32],    # REDUCED for speed
            'kernel_sizes': [3, 3, 3], # REDUCED kernel sizes for speed
            'lstm_units': 16,       # REDUCED from 32 for 2x speedup
            'l2_regularization': 2e-4  # INCREASED to prevent overfitting from faster learning
        },
        'mlp': {
            'learning_rate': 1e-4,
            'batch_size': 48,  # MLPs are more memory efficient
            'dropout1': 0.5,
            'dropout2': 0.3
        },
        'tcn': {
            'learning_rate': 2e-4,   # Increased from 1e-4 for better convergence
            'batch_size': 64,
            'base_filters': 24,      # NEW: Reduced complexity
            'dropout_rate': 0.1,     # NEW: Reduced dropout for better learning
            'dense_units': 32        # NEW: Smaller dense layer
        },
        'dual_branch': {
            'learning_rate': 1e-4,
            'batch_size': 16,
            'dropout1': 0.3,         # INCREASED from 0.3 to fight overfitting
            'dropout2': 0.2,         # INCREASED from 0.2 to fight overfitting
            'kernel_sizes': {
                'ecg': [3, 5],       # Optimized for heart rhythms
                'eeg': [7, 11]       # Optimized for brain waves
            },
            'base_filters': 12,      # REDUCED from 24 to reduce model capacity
            'l2_regularization': 1e-4  # NEW: L2 penalty to prevent overfitting
        },
    }

if __name__ == "__main__":
    # Test the configuration
    print("Testing GPU configuration...")
    
    if gpu_friendly_config():
        print("\n🎯 Recommended settings for your system:")
        print("   • Enable memory growth: Yes")
        print("   • Mixed precision: Yes") 
        print("   • Batch sizes: 16-64 depending on model")
        print("   • Memory limit: 1800MB (leave 200MB buffer)")
        
        # Test memory monitoring
        monitor_gpu_memory()
    else:
        print("❌ GPU configuration failed")