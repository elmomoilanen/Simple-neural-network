from math import isclose

import numpy as np
import pytest

from simple_neural_network.metrics import confusion_matrix, eval_cost


def test_confusion_matrix_binary():
    y_true = np.array(["A", "B", "B", "B"])
    # y_pred is the inverse version
    y_pred = np.array([0, 1, 0, 1])

    corr_cmatrix = np.array([[1, 0], [1, 2]])

    cmatrix = confusion_matrix(y_true, y_pred)

    assert cmatrix.shape == (2, 2)
    assert np.alltrue(corr_cmatrix == cmatrix)


def test_confusion_matrix_multi_category():
    y_true = np.array(["C", "A", "C", "B", "A"])
    # y_pred is the inverse version
    y_pred = np.array([1, 0, 2, 1, 2])

    corr_cmatrix = np.array([[1, 0, 1], [0, 1, 0], [0, 1, 1]])

    cmatrix = confusion_matrix(y_true, y_pred)

    assert cmatrix.shape == (3, 3)
    assert np.alltrue(corr_cmatrix == cmatrix)


def test_confusion_matrix_multi_category_other():
    y_true = np.array(["e", "d", "d", "c", "a", "b", "e", "a", "d", "a", "d"])
    # y_pred is the inverse version
    y_pred = np.array([3, 3, 3, 2, 2, 1, 0, 1, 4, 2, 3])

    corr_cmatrix = np.array(
        [[0, 1, 2, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 3, 1], [1, 0, 0, 1, 0]]
    )

    cmatrix = confusion_matrix(y_true, y_pred)

    assert cmatrix.shape == (5, 5)
    assert np.alltrue(corr_cmatrix == cmatrix)


def test_confusion_matrix_invalid_arguments():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 1]).reshape(-1, 1)

    with pytest.raises(
        ValueError, match=r"Both `y_true` and `y_pred` must be 1-dim arrays, i.e. `y.ndim == 1`"
    ):
        confusion_matrix(y_true, y_pred)

    y_pred = np.array([0, 0, 1])

    with pytest.raises(ValueError, match=r"`y_true` and `y_pred` must be of equal length"):
        confusion_matrix(y_true, y_pred)


def test_eval_cost_reg():
    y_true = np.array([1, -1, 0, 0])
    y_pred = np.array([-1, 1, 0, -2])

    cost = eval_cost(y_true, y_pred, "reg")

    assert isclose(cost, 3.0, abs_tol=0.001)


def test_eval_cost_class():
    y_true = np.array(["a", "b", "a"])
    # y_pred is directly in inverse format
    y_pred = np.array([0, 1, 1])
    y_pred_other = np.array([0, 1, 0])

    cost = eval_cost(y_true, y_pred, "classification")
    cost_other = eval_cost(y_true, y_pred_other, "classification")

    assert cost >= 0 and cost_other >= 0
    assert cost > cost_other
