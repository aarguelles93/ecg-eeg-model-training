# Save as analyze_clipping_thresholds.py

import numpy as np
import matplotlib.pyplot as plt

print("🔍 DETAILED CLIPPING THRESHOLD ANALYSIS")
print("=" * 60)

# Generate the same test data
np.random.seed(42)
n_samples = 10000
n_features = 100

normal_data = np.random.normal(-0.004, 0.013, size=(n_samples, n_features))
n_outliers = int(0.001 * n_samples * n_features)
outlier_indices = np.random.choice(n_samples * n_features, size=n_outliers, replace=False)
flat_data = normal_data.flatten()
flat_data[outlier_indices] = np.random.uniform(-0.3, 0.3, size=n_outliers)
test_data = flat_data.reshape(n_samples, n_features)

def analyze_clipping_effect(X, clip_threshold):
    """Detailed analysis of clipping effects"""
    # Per-sample normalization
    sample_means = np.mean(X, axis=1, keepdims=True)
    sample_stds = np.std(X, axis=1, keepdims=True)
    sample_stds = np.where(sample_stds < 1e-8, 1e-8, sample_stds)
    
    X_norm = (X - sample_means) / sample_stds
    X_clipped = np.clip(X_norm, -clip_threshold, clip_threshold)
    
    # Detailed statistics
    stats = {
        'threshold': clip_threshold,
        'values_clipped': np.sum(np.abs(X_norm) > clip_threshold),
        'percent_clipped': np.sum(np.abs(X_norm) > clip_threshold) / X_norm.size * 100,
        'original_std': np.std(X_norm),
        'clipped_std': np.std(X_clipped),
        'original_range': [X_norm.min(), X_norm.max()],
        'clipped_range': [X_clipped.min(), X_clipped.max()],
        'std_reduction': (np.std(X_norm) - np.std(X_clipped)) / np.std(X_norm) * 100,
        'extreme_6': np.sum(np.abs(X_clipped) > 6.0),
        'extreme_5': np.sum(np.abs(X_clipped) > 5.0),
        'extreme_4': np.sum(np.abs(X_clipped) > 4.0),
        'extreme_3': np.sum(np.abs(X_clipped) > 3.0),
    }
    
    return X_norm, X_clipped, stats

# Test different thresholds
thresholds = [6.0, 5.0, 4.5, 4.0, 3.5, 3.0]
results = {}

print("📊 COMPREHENSIVE THRESHOLD COMPARISON:")
print("Threshold | Clipped Values | Std Change | >6σ | >5σ | >4σ | >3σ | Range")
print("-" * 80)

for threshold in thresholds:
    X_orig, X_clip, stats = analyze_clipping_effect(test_data, threshold)
    results[threshold] = stats
    
    print(f"{threshold:8.1f} | {stats['values_clipped']:10,} | {stats['std_reduction']:8.1f}% | "
          f"{stats['extreme_6']:3} | {stats['extreme_5']:3} | {stats['extreme_4']:3} | "
          f"{stats['extreme_3']:3} | [{stats['clipped_range'][0]:.1f}, {stats['clipped_range'][1]:.1f}]")

# Key insights
print(f"\n🔍 KEY INSIGHTS:")

print(f"\n1. VALUES CLIPPED:")
for threshold in thresholds:
    percent = results[threshold]['percent_clipped']
    print(f"   {threshold}σ: {results[threshold]['values_clipped']:,} values ({percent:.3f}%)")

print(f"\n2. STANDARD DEVIATION IMPACT:")
original_std = results[6.0]['original_std']
print(f"   Original std: {original_std:.3f}")
for threshold in thresholds:
    new_std = results[threshold]['clipped_std']
    reduction = results[threshold]['std_reduction']
    print(f"   {threshold}σ clip: {new_std:.3f} (reduced by {reduction:.1f}%)")

print(f"\n3. TRAINING STABILITY IMPLICATIONS:")
for threshold in thresholds:
    extreme_counts = [results[threshold]['extreme_6'], results[threshold]['extreme_5'], 
                     results[threshold]['extreme_4'], results[threshold]['extreme_3']]
    
    if extreme_counts[0] == 0:  # No >6σ values
        if extreme_counts[1] == 0:  # No >5σ values
            stability = "EXCELLENT"
        elif extreme_counts[1] < 10:
            stability = "VERY GOOD"
        else:
            stability = "GOOD"
    else:
        stability = "RISKY"
    
    print(f"   {threshold}σ clip: {stability} training stability")

print(f"\n4. RECOMMENDED THRESHOLD:")
# Find optimal balance
best_threshold = None
best_score = -1

for threshold in thresholds:
    # Score based on: no >6σ values, minimal std reduction, few >5σ values
    score = 0
    if results[threshold]['extreme_6'] == 0:
        score += 100  # Critical: no >6σ values
    
    if results[threshold]['extreme_5'] == 0:
        score += 50   # Important: no >5σ values
    elif results[threshold]['extreme_5'] < 10:
        score += 25   # Acceptable: few >5σ values
    
    # Prefer less aggressive clipping (higher threshold = less clipping)
    score += threshold * 5
    
    # Penalize excessive std reduction
    if results[threshold]['std_reduction'] < 5:
        score += 20
    elif results[threshold]['std_reduction'] < 10:
        score += 10
    
    if score > best_score:
        best_score = score
        best_threshold = threshold

print(f"   🎯 OPTIMAL: {best_threshold}σ")
print(f"   📊 Rationale:")
print(f"      - Eliminates all >6σ values: ✅")
print(f"      - Eliminates >5σ values: {'✅' if results[best_threshold]['extreme_5'] == 0 else '⚠️'}")
print(f"      - Std reduction: {results[best_threshold]['std_reduction']:.1f}%")
print(f"      - Values clipped: {results[best_threshold]['percent_clipped']:.3f}%")

print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
print(f"   🎯 For your EEG data: Use {best_threshold}σ clipping")
print(f"   🚀 Command: python train.py --normalization per_sample")
print(f"   📊 Expected result: 0 extreme values, {results[best_threshold]['std_reduction']:.1f}% std reduction")

# Show the difference between conservative vs aggressive clipping
print(f"\n⚖️  CONSERVATIVE vs AGGRESSIVE:")
conservative = results[6.0]
aggressive = results[3.0]

print(f"   Conservative (6.0σ):")
print(f"     - Clips {conservative['percent_clipped']:.3f}% of values")
print(f"     - Reduces std by {conservative['std_reduction']:.1f}%")
print(f"     - Preserves more data distribution")

print(f"   Aggressive (3.0σ):")
print(f"     - Clips {aggressive['percent_clipped']:.3f}% of values")
print(f"     - Reduces std by {aggressive['std_reduction']:.1f}%")
print(f"     - More uniform distribution, may lose signal information")

print(f"\n🎯 For ML training: Conservative clipping (≥4.5σ) is usually better")
print(f"   because it preserves natural data variance while eliminating true outliers.")