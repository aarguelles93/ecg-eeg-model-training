# config.py

CONFIG = {
    'global': {
        'epochs': 50,
        'validation_strategy': 'kfold',  # or 'split'
        'n_splits': 5,
        'random_state': 42
    },
    'simple_cnn': {
        'learning_rate': 1e-4,
        'batch_size': 64,
        'dropout': 0.5,
        'filters': [32, 64],
        'kernel_sizes': [3, 3]
    },
    'cnn_lstm': {
        'learning_rate': 2e-4,  # INCREASED for faster convergence
        'batch_size': 32,       # INCREASED from 12 for efficiency
        'dropout': 0.5,         # INCREASED to compensate for faster learning
        'filters': [16, 32],    # REDUCED for speed
        'kernel_sizes': [7, 5, 3],  # 3 kernels for aggressive downsampling
        'lstm_units': 16,       # REDUCED from 32 for 2x speedup
        'l2_regularization': 2e-4
    },
    'mlp': {
        'learning_rate': 1e-4,
        'batch_size': 64,
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
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    }
}
