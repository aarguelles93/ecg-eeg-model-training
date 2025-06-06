# test_cuda_fix.py
# Test script to verify CUDA libdevice path fix

import os

# Set the correct CUDA path BEFORE importing TensorFlow
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
os.environ['CUDA_DIR'] = '/usr/lib/cuda'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress NUMA warnings

print("🔧 Environment variables set:")
print(f"   XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")
print(f"   CUDA_DIR: {os.environ.get('CUDA_DIR', 'Not set')}")

import tensorflow as tf
print(f"📊 TensorFlow version: {tf.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"🖥️  GPUs available: {len(gpus)}")
for gpu in gpus:
    print(f"   {gpu}")

# Test basic training without XLA issues
print("\n🧪 Testing basic model training...")

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create simple test data
X = np.random.random((100, 10))
y = np.random.randint(0, 2, 100)

# Create simple model
model = Sequential([
    Dense(5, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🚀 Starting training test...")
try:
    history = model.fit(X, y, epochs=2, verbose=1, batch_size=16)
    print("✅ SUCCESS! TensorFlow training completed without XLA errors!")
    print(f"   Final accuracy: {history.history['accuracy'][-1]:.3f}")
except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    print("   This suggests the CUDA path fix didn't work completely.")

print("\n🔍 Verifying libdevice file exists...")
libdevice_path = "/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc"
if os.path.exists(libdevice_path):
    print(f"✅ libdevice.10.bc found at: {libdevice_path}")
else:
    print(f"❌ libdevice.10.bc NOT found at: {libdevice_path}")
    
print("\n🎯 If this test passes, your CUDA configuration is fixed!")
print("   You can now run your main training script.")