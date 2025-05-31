import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense,
    Dropout, LSTM, BatchNormalization, Activation,
    Add, LayerNormalization
)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

def build_simple_cnn(input_shape, num_classes):
    assert len(input_shape) == 2, f"Expected 2D input (timesteps, 1), got {input_shape}"

    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_cnn_lstm(input_shape, num_classes):
    assert len(input_shape) == 2, f"Expected 2D input (timesteps, 1), got {input_shape}"

    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_mlp(input_shape, num_classes):
    assert len(input_shape) == 1, f"Expected 1D flattened input, got {input_shape}"
    
    model = Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# === Temporal Convolutional Network (Basic) ===
def tcn_block(inputs, filters, kernel_size, dilation_rate):
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
               padding='causal')(inputs)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
               padding='causal')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)

    # Residual connection
    res = Conv1D(filters, 1, padding='same')(inputs)
    out = Add()([res, x])
    out = Activation('relu')(out)
    return out


def build_tcn(input_shape, num_classes):
    from tensorflow.keras.layers import GlobalAveragePooling1D
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input as KInput

    inputs = KInput(shape=input_shape)
    x = tcn_block(inputs, filters=64, kernel_size=3, dilation_rate=1)
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=2)
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=4)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def build_svm_model(y_train):
    """SVM does not need CNN-like input, it needs flattened feature vectors."""
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    return SVC(kernel='rbf', probability=True, class_weight=class_weight_dict)
