import numpy as np
from sklearn.preprocessing import StandardScaler

# Load raw data and test normalization
data = np.load('data/preprocessed_dataset.npz')
X_train = data['X_train']

print(f"Current data stats:")
print(f"  Mean: {np.mean(X_train):.6f}")
print(f"  Std: {np.std(X_train):.6f}")
print(f"  Range: [{np.min(X_train):.6f}, {np.max(X_train):.6f}]")

# Test normalization
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_train)

print(f"\nAfter normalization:")
print(f"  Mean: {np.mean(X_normalized):.6f}")
print(f"  Std: {np.std(X_normalized):.6f}")
print(f"  Range: [{np.min(X_normalized):.6f}, {np.max(X_normalized):.6f}]")