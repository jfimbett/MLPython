"""
Concise TensorFlow examples aligned with the course slides.

This module includes small demos covering tensors/variables, GradientTape,
low-level training with @tf.function, Keras classification, TensorBoard logging,
and saving a TFLite model. Designed to be short and clear for teaching.

Usage examples (from repository root):

    python src/code/tensorflow_examples.py --demo verify
    python src/code/tensorflow_examples.py --demo gradient_tape
"""
from __future__ import annotations
import argparse
import numpy as np

import tensorflow as tf
from tensorflow import keras
# Access Keras layers through the tensorflow.keras namespace to avoid
# direct submodule import issues in some environments/editors.
layers = keras.layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons


def verify() -> None:
    """Print TensorFlow version and whether a GPU is available.

    Returns: None
    """
    print('TF:', tf.__version__)
    print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)


def tensors_variables() -> None:
    """Show basic constants, random tensors, reductions, and Variables.

    Returns: None
    """
    a = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
    b = tf.random.normal((2, 2))
    print('sum =', tf.reduce_sum(a + b).numpy())
    w = tf.Variable(tf.random.normal((3, 1)))
    print('w shape:', w.shape)


def gradient_tape_demo() -> None:
    """Differentiate a simple scalar function using GradientTape.

    Computes f(x) = x^3 + 2x^2 + 5 at x=2 and prints df/dx.
    Returns: None
    """
    x = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        f = x**3 + 2 * x**2 + 5
    grad = tape.gradient(f, x)
    print('x =', x.numpy(), 'f(x) =', f.numpy(), 'df/dx =', grad.numpy())


def lowlevel_regression_demo() -> None:
    """Low-level training loop using Variables, matmul, and @tf.function.

    Builds a small 3-layer MLP with explicit Variables and trains it using a
    custom train_step with GradientTape and Adam optimizer.
    Returns: None
    """
    # Synthetic regression data
    X = np.linspace(-5, 5, 1000).astype('float32').reshape(-1, 1)
    y = (np.sin(X) + 0.1*np.random.randn(*X.shape)).astype('float32')

    # Model parameters (Variables)
    w1 = tf.Variable(tf.random.normal((1, 64))); b1 = tf.Variable(tf.zeros((64,)))
    w2 = tf.Variable(tf.random.normal((64, 64))); b2 = tf.Variable(tf.zeros((64,)))
    w3 = tf.Variable(tf.random.normal((64, 1))); b3 = tf.Variable(tf.zeros((1,)))
    opt = keras.optimizers.Adam(1e-3)

    @tf.function
    def forward(x):
        x = tf.nn.relu(tf.matmul(x, w1) + b1)
        x = tf.nn.relu(tf.matmul(x, w2) + b2)
        return tf.matmul(x, w3) + b3

    @tf.function
    def train_step(x, y_true):
        with tf.GradientTape() as tape:
            y_pred = forward(x)
            loss = tf.reduce_mean((y_pred - y_true)**2)
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        opt.apply_gradients(zip(grads, [w1, b1, w2, b2, w3, b3]))
        return loss

    for epoch in range(60):
        loss = train_step(X, y)
        if (epoch+1) % 20 == 0:
            tf.print('Epoch', epoch+1, 'Loss', loss)


def keras_binary_classification_demo() -> None:
    """Binary classification with Keras layers inside this TF module.

    Standardizes moons features, trains a small MLP classifier, and prints test
    accuracy.
    Returns: None
    """
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler(); X_train = scaler.fit_transform(X_train); X_test = scaler.transform(X_test)

    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(2,)),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {acc:.3f}')


def tensorboard_demo() -> None:
    """Write TensorBoard logs during a short training run.

    Creates a toy dataset, trains a tiny model for a few epochs with the
    TensorBoard callback, and prints the log directory.
    Returns: None
    """
    import datetime
    X = np.random.randn(500, 4).astype('float32'); y = (X.sum(axis=1) > 0).astype('float32')
    model = keras.Sequential([layers.Dense(8, activation='relu', input_shape=(4,)), layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X, y, epochs=5, batch_size=32, callbacks=[tb_cb], verbose=0)
    print('Logs written to', log_dir)


def save_tflite_demo() -> None:
    """Save a Keras model to .keras and convert to TFLite.

    Builds a tiny model, saves it, converts it to TFLite, and writes the .tflite
    file to disk.
    Returns: None
    """
    model = keras.Sequential([layers.Dense(8, activation='relu', input_shape=(4,)), layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.save('tf_model.keras')
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tfl = conv.convert(); open('model.tflite', 'wb').write(tfl)
    print('Saved tf_model.keras and model.tflite')


DEMOS = {
    'verify': verify,
    'tensors': tensors_variables,
    'gradient_tape': gradient_tape_demo,
    'lowlevel_regression': lowlevel_regression_demo,
    'keras_binary_classification': keras_binary_classification_demo,
    'tensorboard': tensorboard_demo,
    'save_tflite': save_tflite_demo,
}


def main() -> None:
    """CLI entry point to run TensorFlow demos.

    Option:
        --demo  Which demo to run (see DEMOS).

    Returns: None
    """
    parser = argparse.ArgumentParser(description='TensorFlow slide demos')
    parser.add_argument('--demo', choices=DEMOS.keys(), default='verify')
    args = parser.parse_args()
    DEMOS[args.demo]()


if __name__ == '__main__':
    main()
