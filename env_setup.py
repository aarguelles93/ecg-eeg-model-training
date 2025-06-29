#!/usr/bin/env python3
"""
Environment setup for TensorFlow and CUDA
MUST be imported before any TensorFlow-related modules
"""

import os
import warnings

def setup_environment():
    """Set up environment variables for TensorFlow and CUDA before any TF imports"""
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # CRITICAL: TensorFlow memory leak prevention
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # Disable problematic TensorFlow features that cause memory leaks
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
    os.environ['TF_DISABLE_MKL'] = '1'

    # System memory management  
    os.environ['MALLOC_TRIM_THRESHOLD_'] = '0'
    os.environ['MALLOC_MMAP_THRESHOLD_'] = '131072'

    # NUMA suppression  
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # CUDA configuration (critical for your setup)
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
    os.environ['CUDA_DIR'] = '/usr/lib/cuda'
    
    print("🔧 Environment variables set for TensorFlow/CUDA compatibility")
    return True

# Auto-setup when imported
setup_environment()