"""Implements metrics to evaluate performance of neural networks."""
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Contingency table between true (rows) and predicted (columns) values of y.

    Use this metric only for classification type tasks.

    Pass arrays in their original value labels (numerical, categorical) or
    in their inverse format. E.g., the inverse of ["a", "b"] is [0, 1].

    Params
    ------
    y_true: NumPy array
        True/actual y with numerical or categorical values of shape (n,).

    y_pred: NumPy array
        Predicted y with numerical or categorical values of shape (n,).

    Returns
    -------
    NumPy array
        Rows represent the true labels with zero-based indexing and columns
        corresponding predicted labels. E.g. entry [0, 1] represents the case where the
        true label is zero and predicted one. Thus, diagonal entries indicates the
        correctly predicted counts.
    """
    if not (y_true.ndim == 1 and y_pred.ndim == 1):
        raise ValueError("Both `y_true` and `y_pred` must be 1-dim arrays, i.e. `y.ndim == 1`")

    if y_true.shape != y_pred.shape:
        raise ValueError("`y_true` and `y_pred` must be of equal length")

    y_uniq, y_true_inv = np.unique(y_true, return_inverse=True)
    _, y_pred_inv = np.unique(y_pred, return_inverse=True)

    dim = y_uniq.shape[0]
    matrix = np.zeros((dim, dim), dtype=np.int64)

    for j in range(y_true_inv.shape[0]):
        matrix[y_true_inv[j], y_pred_inv[j]] += 1

    return matrix


def eval_cost(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    """Evaluate cost of the neural network model.

    Params
    ------
    y_true: NumPy array
        True/actual y with numerical or categorical values of shape (n,).

    y_pred: NumPy array
        Predicted y with numerical or categorical values of shape (n,).

    method: str
        Type of learning, either `class` for classification or `reg` for regression.

    Returns
    -------
    float
        Cross entropy for classification type learning tasks, mean square error (MSE)
        for regression tasks.
    """
    if not (y_true.ndim == 1 and y_pred.ndim == 1):
        raise ValueError("Both `y_true` and `y_pred` must be 1-dim arrays, i.e. `y.ndim == 1`")

    if y_true.shape != y_pred.shape:
        raise ValueError("`y_true` and `y_pred` must be of equal length")

    if method in ("class", "classification"):
        _, y_true_inv = np.unique(y_true, return_inverse=True)
        _, y_pred_inv = np.unique(y_pred, return_inverse=True)

        # turn boolean values to integers by multiplying with one
        matches = (y_true_inv == y_pred_inv) * 1
        log_y = np.log(matches + 1e-9)

        # don't allow zero or small negative return values
        # e.g. if `matches` array contains only ones, the return value would be slightly negative
        return max(-np.mean(log_y), 1e-20)

    if method not in ("reg", "regression"):
        raise ValueError(f"Cannot evaluate cost for learning type `{method}`")

    square_of_err = (y_true - y_pred) ** 2
    return np.mean(square_of_err)
