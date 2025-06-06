import numpy as np
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

# UPDATED: Import from training_utils
from training_utils import create_optimizer, cleanup_memory

# === Simple CNN ===
def build_simple_cnn(input_shape, num_classes, config=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
    )

    config = config or {}
    filters = config.get('filters', [32, 64])
    kernel_sizes = config.get('kernel_sizes', [3, 3])
    dropout = config.get('dropout', 0.5)

    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters[0], kernel_size=kernel_sizes[0], activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters[1], kernel_size=kernel_sizes[1], activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid') if num_classes == 2 else Dense(num_classes, activation='softmax')
    ])
    return model

# === CNN + LSTM ===
def build_cnn_lstm(input_shape, num_classes, config=None):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
    )
    from tensorflow.keras.regularizers import l2

    config = config or {}
    filters = config.get('filters', [16, 32])
    kernel_sizes = config.get('kernel_sizes', [7, 5, 3])  # 3 kernels for 3 conv layers
    dropout = config.get('dropout', 0.5)
    
    # SPEED OPTIMIZED: Even smaller LSTM
    lstm_units = config.get('lstm_units', 16)
    l2_reg = config.get('l2_regularization', 2e-4)

    print(f"[DEBUG] CNN-LSTM input_shape received: {input_shape}")
    if len(input_shape) != 2:
        print("[WARNING] Expected 2D input shape (timesteps, features) for LSTM compatibility.")
    else:
        print(f"[INFO] Assuming format: (timesteps={input_shape[0]}, features={input_shape[1]})")

    reg = l2(l2_reg) if l2_reg > 0 else None

    model = Sequential([
        Input(shape=input_shape),
        # AGGRESSIVE DOWNSAMPLING: Reduce sequence length dramatically
        Conv1D(filters[0], kernel_size=kernel_sizes[0], activation='relu'),  # Use config kernel
        MaxPooling1D(pool_size=4),  # More aggressive pooling: 6016 → 1503
        
        Conv1D(filters[1], kernel_size=kernel_sizes[1], activation='relu'),  # Use config kernel
        MaxPooling1D(pool_size=4),  # More aggressive pooling: 1503 → 375
        
        Conv1D(filters[1], kernel_size=kernel_sizes[2], activation='relu'),  # Use config kernel
        MaxPooling1D(pool_size=4),  # Final reduction: 375 → 93
        
        # NOW LSTM processes only ~93 timesteps instead of 1504!
        LSTM(lstm_units,
             activation='tanh',
             recurrent_activation='sigmoid',
             dropout=0.0,
             recurrent_dropout=0.0,
             unroll=False,
             use_bias=True),
        
        # Smaller dense layers for speed
        Dense(16, activation='relu', kernel_regularizer=reg),  # Reduced from 32
        Dropout(dropout),
        Dense(1, activation='sigmoid') if num_classes == 2 else Dense(num_classes, activation='softmax')
    ])
    return model

# === MLP ===
def build_mlp(input_shape, num_classes=2, config=None):
    """
    Build memory-efficient MLP model
    ENHANCED with optimized architecture for GTX 1050
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    if config is None:
        config = {
            'learning_rate': 1e-4,
            'batch_size': 48,      # Optimized for GTX 1050
            'dropout1': 0.5,
            'dropout2': 0.3
        }
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),    # Optimized size
        Dropout(config.get('dropout1', 0.5)),
        Dense(32, activation='relu'),                            # Reduced for memory
        Dropout(config.get('dropout2', 0.3)),
        Dense(1 if num_classes == 2 else num_classes, 
              activation='sigmoid' if num_classes == 2 else 'softmax')
    ])
    
    return model

# === TCN Block ===
def tcn_block(inputs, filters, kernel_size, dilation_rate, dropout_rate=0.1):
    from tensorflow.keras.layers import (
        Conv1D, BatchNormalization, Activation, Dropout, Add
    )
    from tensorflow.keras.regularizers import l2

    # Added regularization
    reg = l2(1e-4)
    
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, 
               padding='causal', kernel_regularizer=reg)(inputs)  
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    # ADDITIONAL RESIDUAL CONNECTION
    res = Conv1D(filters, 1, padding='same', kernel_regularizer=reg)(inputs)
    out = Add()([res, x])
    out = Activation('relu')(out)
    return out

def build_tcn(input_shape, num_classes, config=None):
    from tensorflow.keras.layers import Input as KInput, Dense, Dropout, GlobalAveragePooling1D, GaussianNoise
    from tensorflow.keras import Model
    from tensorflow.keras.regularizers import l2

    config = config or {}

    base_filters = config.get('base_filters', 16)
    dropout_rate = config.get('dropout_rate', 0.3)
    dense_units = config.get('dense_units', 32)
    
    inputs = KInput(shape=input_shape)
    
    # ANTI-OVERFITTING: Added Gaussian noise layer
    x = GaussianNoise(0.01)(inputs)
    
    x = tcn_block(x, filters=base_filters, kernel_size=3, dilation_rate=1, dropout_rate=dropout_rate)
    x = tcn_block(x, filters=base_filters, kernel_size=3, dilation_rate=2, dropout_rate=dropout_rate)
    x = tcn_block(x, filters=base_filters, kernel_size=3, dilation_rate=4, dropout_rate=dropout_rate)  # Added
    
    x = GlobalAveragePooling1D()(x)
    # ANTI-OVERFITTING: Reduced dense layer size
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)

    outputs = Dense(1, activation='sigmoid')(x) if num_classes == 2 else Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# === SVM ===
def build_svm_model(y_train, config=None):
    config = config or {}
    kernel = config.get('kernel', 'rbf')
    C = config.get('C', 1.0)
    gamma = config.get('gamma', 'scale')
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    return SVC(kernel=kernel, C=C, gamma=gamma, probability=True, class_weight=class_weight_dict)

# === Dual-Branch CNN ===
def build_dual_branch_cnn(input_shape, num_classes, config=None):
    from tensorflow.keras.layers import (
        Input, Conv1D, BatchNormalization, GlobalMaxPooling1D, 
        GlobalAveragePooling1D, Concatenate, Dense, Dropout
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras import Model

    config = config or {}
    kernel_sizes = config.get('kernel_sizes', {'ecg': [3, 5], 'eeg': [7, 11]})
    dropout1 = config.get('dropout1', 0.5)
    dropout2 = config.get('dropout2', 0.4)
    
    # ANTI-OVERFITTING: Reduced model capacity
    base_filters = config.get('base_filters', 16)
    l2_reg = config.get('l2_regularization', 1e-3)
    
    # Create regularizer
    reg = l2(l2_reg) if l2_reg > 0 else None

    signal_input = Input(shape=input_shape, name='signal_input')

    # ECG-optimized branch (shorter kernels for heart rhythm patterns)
    ecg_branch = Conv1D(base_filters, kernel_size=kernel_sizes['ecg'][0], activation='relu', kernel_regularizer=reg)(signal_input)
    # ecg_branch = BatchNormalization()(ecg_branch)
    ecg_branch = Conv1D(base_filters * 2, kernel_size=kernel_sizes['ecg'][1], activation='relu', kernel_regularizer=reg)(ecg_branch)
    ecg_branch = GlobalMaxPooling1D()(ecg_branch)

    # EEG-optimized branch (longer kernels for brain wave patterns)
    eeg_branch = Conv1D(base_filters, kernel_size=kernel_sizes['eeg'][0], activation='relu', kernel_regularizer=reg)(signal_input)
    # eeg_branch = BatchNormalization()(eeg_branch)
    eeg_branch = Conv1D(base_filters * 2, kernel_size=kernel_sizes['eeg'][1], activation='relu', kernel_regularizer=reg)(eeg_branch)
    eeg_branch = GlobalAveragePooling1D()(eeg_branch)

    # ANTI-OVERFITTING: Smaller dense layers with stronger regularization
    merged = Concatenate()([ecg_branch, eeg_branch])
    merged = Dense(8, activation='relu', kernel_regularizer=reg)(merged)
    merged = Dropout(dropout1)(merged)
    merged = Dense(4, activation='relu', kernel_regularizer=reg)(merged)
    merged = Dropout(dropout2)(merged)

    output = Dense(1, activation='sigmoid')(merged) if num_classes == 2 else Dense(num_classes, activation='softmax')(merged)
    model = Model(inputs=signal_input, outputs=output)
    return model
