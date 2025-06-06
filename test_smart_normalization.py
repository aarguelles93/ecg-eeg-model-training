import os
import sys
import numpy as np

# Add CUDA fix
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
os.environ['CUDA_DIR'] = '/usr/lib/cuda'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("🧪 Testing Smart Normalization Fix")
print("=" * 50)

# Test 1: Verify CUDA path
libdevice_path = "/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc"
if os.path.exists(libdevice_path):
    print("✅ CUDA libdevice found at correct path")
else:
    print("❌ CUDA libdevice NOT found - check path")

# Test 2: Import your functions
try:
    sys.path.append('.')  # Add current directory
    from data_loader import detect_existing_normalization, smart_normalize_single_dataset
    print("✅ Smart normalization functions imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you added the functions to data_loader.py")

# Test 3: Test smart normalization logic
print("\n🧪 Testing normalization detection...")

# Create test data that mimics your ECG (already min-max normalized)
ecg_test = np.random.uniform(0, 1, size=(1000, 100))  # Min-max normalized data
print(f"📊 ECG test data: range=[{ecg_test.min():.3f}, {ecg_test.max():.3f}], mean={ecg_test.mean():.3f}")

# Create test data that mimics your EEG (appears to be in reasonable range)
eeg_test = np.random.normal(-0.004, 0.013, size=(1000, 100))  # Similar to your EEG stats
print(f"📊 EEG test data: range=[{eeg_test.min():.3f}, {eeg_test.max():.3f}], mean={eeg_test.mean():.3f}")

# Test detection
try:
    ecg_status = detect_existing_normalization(ecg_test, "ECG_TEST")
    eeg_status = detect_existing_normalization(eeg_test, "EEG_TEST")
    
    print(f"\n📈 Detection results:")
    print(f"   ECG status: {ecg_status}")
    print(f"   EEG status: {eeg_status}")
    
    # Test smart normalization
    ecg_norm, _ = smart_normalize_single_dataset(ecg_test, "ECG_TEST")
    eeg_norm, _ = smart_normalize_single_dataset(eeg_test, "EEG_TEST")
    
    print(f"\n📊 After smart normalization:")
    print(f"   ECG: range=[{ecg_norm.min():.3f}, {ecg_norm.max():.3f}], mean={ecg_norm.mean():.3f}")
    print(f"   EEG: range=[{eeg_norm.min():.3f}, {eeg_norm.max():.3f}], mean={eeg_norm.mean():.3f}")
    
    # Check for extreme values
    ecg_extreme = np.sum(np.abs(ecg_norm) > 6)
    eeg_extreme = np.sum(np.abs(eeg_norm) > 6)
    
    if ecg_extreme == 0 and eeg_extreme == 0:
        print("✅ No extreme values detected - smart normalization working!")
    else:
        print(f"❌ Still found extreme values: ECG={ecg_extreme}, EEG={eeg_extreme}")
        
except Exception as e:
    print(f"❌ Error testing smart normalization: {e}")

# Test 4: Quick TensorFlow test
print("\n🧪 Testing TensorFlow with CUDA fix...")
try:
    import tensorflow as tf
    print(f"📊 TensorFlow version: {tf.__version__}")
    
    # Quick model test
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    X = np.random.random((100, 10))
    y = np.random.randint(0, 2, 100)
    
    model = Sequential([Dense(5, activation='relu'), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("🚀 Training quick test model...")
    model.fit(X, y, epochs=1, verbose=0, batch_size=16)
    print("✅ TensorFlow test passed - no XLA errors!")
    
except Exception as e:
    print(f"❌ TensorFlow test failed: {e}")

print("\n🎯 Test Summary:")
print("   If all tests passed, you can now run:")
print("   python train.py svm --normalization smart --quick-lc")
print("\n   This should fix the extreme value issue!")