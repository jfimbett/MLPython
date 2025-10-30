"""
Hands-on Keras (TensorFlow) examples to match the course slides.

This module provides small, focused demos for common deep learning workflows:
regression, binary and multiclass classification, callbacks, regularization, and
model persistence. 

Usage examples (from repository root):

    python src/code/keras_examples.py --demo verify
    python src/code/keras_examples.py --demo regression

Notes:
- Some demos support optional plotting via the `show_plot` flag.
- See the DEMOS dict for all available keys.
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_blobs

import os
import platform
import importlib.util

# Choose a safe Keras backend before importing keras to avoid TF AVX aborts.
# Policy:
# - Respect user-provided KERAS_BACKEND if set.
# - On macOS (especially Intel without AVX or Apple Silicon without proper TF build),
#   default to 'torch' backend to avoid TensorFlow CPU instruction issues.
# - On Windows/Linux, default to 'tensorflow'. Users can override via env var.
if 'KERAS_BACKEND' not in os.environ:
    system = platform.system()
    machine = platform.machine().lower()
    if system == 'Darwin':
        # Safer default on macOS is torch (works on Intel and Apple Silicon).
        os.environ['KERAS_BACKEND'] = 'torch'
    else:
        os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras  # Keras 3 multi-backend frontend
from keras import layers, regularizers


def verify() -> None:
    """Print installed Keras version, chosen backend, and backend version.

    This avoids importing TensorFlow unless it's the active backend to prevent
    AVX-related aborts on some macOS/CPU setups.

    Returns: None
    """
    backend = os.environ.get('KERAS_BACKEND', 'tensorflow')
    print('Keras:', keras.__version__)
    print('Keras backend:', backend)
    if backend == 'tensorflow':
        try:
            import tensorflow as tf  # Local import to avoid abort unless needed
            print('TensorFlow:', tf.__version__)
        except Exception as e:
            print('TensorFlow import failed:', e)
    elif backend == 'torch':
        try:
            import torch
            print('PyTorch:', torch.__version__)
        except Exception as e:
            print('PyTorch import failed:', e)


def regression(show_plot: bool = True) -> None:
    """Nonlinear regression with a small MLP in Keras.

    Creates synthetic 1D inputs and a nonlinear target (sin/cos mix with noise),
    trains a small feed-forward network, reports test MAE, and optionally plots
    training curves (loss and MAE).

    Args:
        show_plot: If True, show training/validation curves.

    Returns: None
    """
    # Make synthetic data
    X = np.linspace(-5, 5, 1000).reshape(-1, 1)
    y = np.sin(X) + np.cos(2 * X) + np.random.randn(1000, 1) * 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a simple MLP model for regression
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train with validation split for monitoring
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate on held-out test data
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test MAE: {test_mae:.4f}')

    # Optionally visualize training curves
    if show_plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train')
        plt.plot(history.history['val_mae'], label='Val')
        plt.title('MAE'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.show()


def binary_classification() -> None:
    """Binary classification on moons dataset with regularization.

    Standardizes features, defines a small MLP with dropout, trains with early
    stopping, and reports test accuracy.

    Returns: None
    """
    # Generate and split data
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features for faster/more stable training
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define a regularized classifier
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(2,)),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping to avoid overfitting and reduce training time
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)

    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {acc:.4f}')


def multiclass_classification() -> None:
    """Multiclass classification on synthetic blobs.

    Standardizes features, trains a small softmax classifier, and prints test
    accuracy. Uses sparse categorical cross-entropy with integer labels.

    Returns: None
    """
    # Generate and split multiclass data
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=1.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define a simple softmax classifier
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(2,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate on test data
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {acc:.4f}')


def callbacks_demo() -> None:
    """Demonstrate EarlyStopping and ReduceLROnPlateau callbacks.

    Trains a small regression MLP with synthetic data using callbacks that stop
    early when validation loss stalls and reduce learning rate on plateaus.

    Returns: None
    """
    # Synthetic cubic-ish regression data
    X = np.linspace(-2, 2, 400).reshape(-1, 1)
    y = (X**3 - 0.5*X + 0.1*np.random.randn(*X.shape)).astype('float64')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')

    # Useful training-time callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ]
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=0)

    print('Val loss (best):', min(model.history.history['val_loss']))


def regularization_demo() -> None:
    """Showcase L2/L1L2 regularization and dropout in a small model.

    Builds a simple network with L2 and L1L2 kernel regularizers and dropout
    layers, compiles it, and prints a model summary.

    Returns: None
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-2), input_shape=(10,)),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(1e-3, 1e-3)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()


def save_load_demo() -> None:
    """Save and load a tiny Keras model to .keras format.

    Creates, compiles, and saves a model; then loads it back and checks type.

    Returns: None
    """
    # Tiny model
    model = keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=(4,)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.save('my_model.keras')
    loaded = keras.models.load_model('my_model.keras')
    print('Loaded model ok:', isinstance(loaded, keras.Model))


DEMOS = {
    'verify': verify,
    'regression': regression,
    'binary_classification': binary_classification,
    'multiclass': multiclass_classification,
    'callbacks': callbacks_demo,
    'regularization': regularization_demo,
    'save_load': save_load_demo,
}


def main() -> None:
    """CLI entry point to run Keras demos.

    Options:
        --demo     Which demo to run (see DEMOS).
        --no-plot  For 'regression', disables plotting.

    Returns: None
    """
    parser = argparse.ArgumentParser(description='Keras slide demos')
    parser.add_argument('--demo', choices=DEMOS.keys(), default='verify')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    if args.demo == 'regression' and args.no_plot:
        regression(show_plot=False)
    else:
        DEMOS[args.demo]()


if __name__ == '__main__':
    main()
