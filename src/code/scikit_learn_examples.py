"""
Hands-on scikit-learn examples that match the course slides.

This module exposes small, focused demo functions you can run from the CLI to
illustrate common ML workflows such as prediction, training/test splits,
evaluation metrics, feature scaling, decision boundaries, and basic tree-based
models. Each demo prints a compact summary and, for some, optionally displays a
plot to visualize the result.

Usage examples (run from repository root):

    python src/code/scikit_learn_examples.py --demo verify
    python src/code/scikit_learn_examples.py --demo linear_regression
    python src/code/scikit_learn_examples.py --demo decision_boundary

Notes:
- Use --no-plot to disable plots for demos that support it.
- See the DEMOS mapping at the bottom of this file for available choices.
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def verify() -> None:
    """Quick environment sanity check.

    Imports scikit-learn to verify availability and prints the installed
    version. Useful to confirm your Python environment is correctly set up
    before running other demos.

    Returns: None
    """
    import sklearn  # Local import to ensure the package exists at runtime
    print(f"Scikit-learn version: {sklearn.__version__}")


def linear_regression_demo(show_plot: bool = True) -> None:
    """Fit and visualize a simple univariate linear regression.

    This demo synthesizes data from the linear relation y = 2x + 1 plus Gaussian
    noise, trains a LinearRegression model, prints the learned coefficient and
    intercept, and optionally plots the fitted line against the noisy data.

    Args:
        show_plot: If True, display a matplotlib scatter + line plot.

    Returns: None
    """
    # Create synthetic data: y = 2x + 1 + noise
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.5

    # Define and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Predict on the training grid for visualization
    y_pred = model.predict(X)

    # Report parameters of the fitted model
    print(f"Coefficient: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")

    # Optional visualization: data points and fitted line
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.5, label='Data')
        plt.plot(X, y_pred, 'r-', linewidth=2, label='Prediction')
        plt.xlabel('X'); plt.ylabel('y'); plt.legend()
        plt.title('Linear Regression Example'); plt.grid(True, alpha=0.3)
        plt.show()


def polynomial_regression_demo(show_plot: bool = True) -> None:
    """Approximate a nonlinear function using polynomial features.

    This demo approximates y ≈ sin(x) by expanding inputs with
    PolynomialFeatures(degree=5) and fitting a LinearRegression on top. It shows
    how polynomial basis expansion allows linear models to capture nonlinearity.

    Args:
        show_plot: If True, display a plot comparing predictions to noisy data.

    Returns: None
    """
    # Generate nonlinear target: y ~ sin(x) with noise
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.randn(100) * 0.1

    # Build a pipeline: polynomial expansion -> linear regression
    poly_model = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
    poly_model.fit(X, y)

    # Predict across the grid for visualization
    y_poly_pred = poly_model.predict(X)

    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, s=16, alpha=0.6, label='Data')
        plt.plot(X, y_poly_pred, 'r-', linewidth=2, label='Poly deg=5')
        plt.xlabel('X'); plt.ylabel('y'); plt.legend()
        plt.title('Polynomial Regression (deg=5)'); plt.grid(True, alpha=0.3)
        plt.show()


def train_test_split_and_metrics_demo() -> None:
    """Demonstrate train/test split and regression metrics.

    Creates synthetic linear data with 5 features, splits into train/test sets,
    fits a LinearRegression model, and reports the Mean Squared Error (MSE) and
    R-squared (R2) on the test set.

    Returns: None
    """
    # Generate synthetic linear data with 5 features
    X = np.random.randn(1000, 5)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(1000) * 0.5

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit a simple linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"R2:  {r2:.4f}")


def logistic_classification_demo() -> None:
    """Binary classification with Logistic Regression and common metrics.

    Builds a small synthetic 2D dataset, trains a logistic regression classifier,
    computes accuracy, precision, recall, F1-score, and prints the confusion
    matrix on a held-out test set.

    Returns: None
    """
    # Generate a small, linearly separable-ish dataset
    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1,
        random_state=42
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Predict and compute metrics
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


def decision_boundary_plot_demo() -> None:
    """Visualize a 2D decision boundary for logistic regression.

    Generates a small 2D dataset, fits LogisticRegression, evaluates predictions
    on a dense meshgrid, and plots a filled contour to show the decision
    regions along with the training points.

    Returns: None
    """
    # Reuse a small 2D classification dataset
    X, y = make_classification(
        n_samples=300, n_features=2, n_redundant=0,
        n_informative=2, n_clusters_per_class=1,
        random_state=42
    )
    clf = LogisticRegression().fit(X, y)

    # Create a grid that covers the data range
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Predict across the grid and reshape to image
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot decision regions and data points
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title('Decision Boundary (Logistic Regression)')
    plt.show()


def cross_validation_demo() -> None:
    """Evaluate a classifier with k-fold cross-validation.

    Creates a small dataset and uses 5-fold cross-validation to estimate the
    accuracy of a LogisticRegression classifier. Prints per-fold scores and
    their mean and standard deviation.

    Returns: None
    """
    # Generate data and evaluate with cross-validation
    X, y = make_classification(
        n_samples=500, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=42
    )
    clf = LogisticRegression()
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("CV scores:", scores)
    print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")


def feature_scaling_demo() -> None:
    """Show the impact of feature scaling on model performance.

    Generates a classification dataset, splits it, standardizes features using
    StandardScaler (fit on train, transform train/test), trains Logistic
    Regression, and prints the test accuracy. Demonstrates a typical preprocessing
    workflow.

    Returns: None
    """
    # Generate data and split
    X, y = make_classification(
        n_samples=400, n_features=4, n_redundant=0, n_informative=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit scaler on training data, then transform train and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate
    clf = LogisticRegression().fit(X_train_scaled, y_train)
    print(f"Accuracy with scaling: {clf.score(X_test_scaled, y_test):.3f}")


def decision_tree_demo() -> None:
    """Train a shallow decision tree and report accuracy.

    Builds a simple classification dataset, fits a DecisionTreeClassifier with a
    bounded depth (to reduce overfitting), and prints training and test accuracy
    to highlight generalization.

    Returns: None
    """
    # Generate data and split
    X, y = make_classification(
        n_samples=400, n_features=4, n_redundant=0, n_informative=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit a shallow tree
    tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_clf.fit(X_train, y_train)

    # Compare train vs test accuracy
    print(f"Train acc: {tree_clf.score(X_train, y_train):.3f}")
    print(f"Test  acc: {tree_clf.score(X_test, y_test):.3f}")


def random_forest_demo() -> None:
    """Fit a Random Forest and inspect test accuracy and feature importances.

    Trains a RandomForestClassifier on synthetic data, evaluates test accuracy,
    and prints the model's feature importances to illustrate how ensembles can
    provide interpretability of feature contributions.

    Returns: None
    """
    # Create a moderately sized dataset and split
    X, y = make_classification(
        n_samples=600, n_features=6, n_redundant=0, n_informative=3,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a random forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate and report feature importances
    print(f"RF test accuracy: {rf.score(X_test, y_test):.3f}")
    print("Feature importances:", rf.feature_importances_)


DEMOS = {
    'verify': verify,
    'linear_regression': linear_regression_demo,
    'polynomial_regression': polynomial_regression_demo,
    'split_and_metrics': train_test_split_and_metrics_demo,
    'logistic_classification': logistic_classification_demo,
    'decision_boundary': decision_boundary_plot_demo,
    'cross_validation': cross_validation_demo,
    'feature_scaling': feature_scaling_demo,
    'decision_tree': decision_tree_demo,
    'random_forest': random_forest_demo,
}


def main() -> None:
    """Command-line interface to run the available demos.

    Options:
        --demo     Name of the demo to run (see DEMOS keys).
        --no-plot  For demos that support plotting, disable figure display.

    Behavior:
        If --no-plot is passed for demos that accept a show_plot flag, the demo
        is invoked with show_plot=False. Otherwise it is called with defaults.

    Returns: None
    """
    parser = argparse.ArgumentParser(description='Scikit-learn slide demos')
    parser.add_argument('--demo', choices=DEMOS.keys(), default='verify')
    parser.add_argument('--no-plot', action='store_true', help='Disable plots')
    args = parser.parse_args()

    func = DEMOS[args.demo]
    if args.no_plot and func in (linear_regression_demo, polynomial_regression_demo):
        func(show_plot=False)
    else:
        func()


if __name__ == '__main__':
    main()
