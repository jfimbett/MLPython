"""
Hands-on PyTorch examples aligned with the course slides.

This module contains compact demos that showcase tensor basics, autograd,
derivatives, simple regression/classification models, data loaders, scalar
optimization, and saving/loading weights. Each function is short and designed
for live instruction.

Usage examples (from repository root):

    python src/code/pytorch_examples.py --demo verify
    python src/code/pytorch_examples.py --demo autograd_scalar
    python src/code/pytorch_examples.py --demo regression
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


def verify() -> None:
    """Print PyTorch version and CUDA availability.

    Quick environment check before running other demos.
    Returns: None
    """
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())


def tensors_basics() -> None:
    """Show basic tensor creation, shapes, dtypes and simple reductions.

    Returns: None
    """
    x = torch.tensor([1, 2, 3, 4, 5])
    y = torch.zeros(3, 4)
    z = torch.ones(2, 3)
    rnd = torch.randn(2, 3)
    print('x shape:', x.shape, 'dtype:', x.dtype)
    print('z sum:', z.sum().item(), 'rnd mean:', rnd.mean().item())


def autograd_scalar() -> None:
    """Differentiate a scalar function using autograd.

    Computes y = x^3 + 2x^2 + 5 at x=2 and prints dy/dx via backward().
    Returns: None
    """
    x = torch.tensor([2.0], requires_grad=True)
    y = x**3 + 2*x**2 + 5
    y.backward()
    print('x:', x.item(), 'y:', y.item(), 'dy/dx:', x.grad.item())  # 20


def derivative_plot() -> None:
    """Plot a function and its derivative computed via autograd.

    Uses a differentiable tensor to compute f(x) = sin(x) + x^2/10 and its
    derivative, then plots both.
    Returns: None
    """
    x = torch.linspace(-5, 5, 100, requires_grad=True)
    f = torch.sin(x) + x**2 / 10
    f.sum().backward()
    df_dx = x.grad
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(x.detach().numpy(), f.detach().numpy()); plt.title('f(x)')
    plt.subplot(1, 2, 2); plt.plot(x.detach().numpy(), df_dx.numpy()); plt.title("f'(x)")
    plt.show()


class RegressionNet(nn.Module):
    """Small MLP for 1D regression: 1->64->64->32->1 with ReLU activations."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass returning regression output."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


def regression_demo() -> None:
    """Train a small MLP for nonlinear 1D regression and report loss.

    Synthesizes sin/cos data with noise, trains RegressionNet using Adam on MSE,
    prints periodic training loss and final test loss.
    Returns: None
    """
    # Prepare synthetic regression data
    X = np.linspace(-5, 5, 1000).reshape(-1, 1)
    y = np.sin(X) + np.cos(2*X) + np.random.randn(1000, 1) * 0.1
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Model, loss, optimizer
    model = RegressionNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Simple training loop
    for epoch in range(50):
        out = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1:03d} loss {loss.item():.4f}')

    # Evaluate on test split
    model.eval()
    with torch.no_grad():
        test_loss = criterion(model(X_test), y_test)
        print('Test loss:', float(test_loss))


class Classifier(nn.Module):
    """Binary classifier MLP with dropout and sigmoid output."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass returning probabilities in [0,1]."""
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.sigmoid(self.fc3(x))


def binary_classification_demo() -> None:
    """Binary classification on moons dataset with PyTorch.

    Standardizes features, trains the Classifier with BCE loss and Adam, prints
    periodic training metrics, and evaluates accuracy and reports confusion
    matrix/classification report on the test set.
    Returns: None
    """
    # Prepare data
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    scaler = StandardScaler(); X = scaler.fit_transform(X)
    X_t = torch.FloatTensor(X); y_t = torch.FloatTensor(y).view(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

    # Model, loss, optimizer
    model = Classifier(); criterion = nn.BCELoss(); optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(60):
        out = model(X_train); loss = criterion(out, y_train)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch+1) % 20 == 0:
            pred = (out > 0.5).float(); acc = (pred == y_train).float().mean().item()
            print(f'Epoch {epoch+1:03d} loss {loss.item():.4f} acc {acc:.3f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(X_test); test_pred = (test_out > 0.5).float()
        acc = (test_pred == y_test).float().mean().item()
        print('Test accuracy:', round(acc, 3))
        print('Confusion matrix:\n', confusion_matrix(y_test.numpy(), test_pred.numpy()))
        print('Report:\n', classification_report(y_test.numpy(), test_pred.numpy()))


def dataloader_demo() -> None:
    """Demonstrate DataLoader for batching/shuffling.

    Creates a random regression dataset, wraps it in TensorDataset/DataLoader,
    and runs a single training epoch over batches.
    Returns: None
    """
    X = torch.randn(1000, 10); y = torch.randn(1000, 1)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.MSELoss()
    for xb, yb in dl:
        loss = loss_fn(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    print('One epoch done with DataLoader.')


def compute_gradient_func() -> None:
    """Compute derivative of a scalar function at a point using autograd.

    Defines a helper that takes a callable f and a float value x, and returns
    df/dx using PyTorch autograd. Prints examples for x^2 and sin(x).
    Returns: None
    """
    def compute_gradient(func, x_val: float) -> float:
        x = torch.tensor([x_val], requires_grad=True)
        y = func(x); y.backward(); return x.grad.item()
    print('d/dx x^2 at 3   =', compute_gradient(lambda t: t**2, 3.0))
    print('d/dx sin(x) at 0=', compute_gradient(torch.sin, 0.0))


def optimize_scalar_demo() -> None:
    """Use SGD to minimize a simple 1D function.

    Minimizes (x-3)^2 + 5 starting from x=0 using gradient descent.
    Returns: None
    """
    x = torch.tensor([0.0], requires_grad=True)
    opt = optim.SGD([x], lr=0.1)
    for i in range(60):
        opt.zero_grad(); y = (x - 3)**2 + 5; y.backward(); opt.step()
    print('Minimum near x=', round(x.item(), 4))


def save_load_demo() -> None:
    """Save and load a model's state_dict from disk.

    Demonstrates basic PyTorch persistence using torch.save/torch.load.
    Returns: None
    """
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    torch.save(model.state_dict(), 'model_weights.pth')
    loaded = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    loaded.load_state_dict(torch.load('model_weights.pth'))
    loaded.eval(); print('Weights loaded ok')


DEMOS = {
    'verify': verify,
    'tensors': tensors_basics,
    'autograd_scalar': autograd_scalar,
    'derivative_plot': derivative_plot,
    'regression': regression_demo,
    'binary_classification': binary_classification_demo,
    'dataloader': dataloader_demo,
    'compute_gradient': compute_gradient_func,
    'optimize_scalar': optimize_scalar_demo,
    'save_load': save_load_demo,
    'regression_demo': regression_demo
}


def main() -> None:
    """CLI entry point to run PyTorch demos.

    Option:
        --demo  Which demo to run (see DEMOS).

    Returns: None
    """
    parser = argparse.ArgumentParser(description='PyTorch slide demos')
    parser.add_argument('--demo', choices=DEMOS.keys(), default='verify')
    args = parser.parse_args()
    DEMOS[args.demo]()


if __name__ == '__main__':
    main()
